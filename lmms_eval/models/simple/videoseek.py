import json
import multiprocessing as mp
import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.videoseek_native import VENDORED_UPSTREAM_COMMIT


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _subtitle_path_for_video(video_path: str) -> str | None:
    path = Path(video_path)
    candidates = [path.with_suffix(".srt"), path.parent.parent / "subtitle" / f"{path.stem}.srt"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _build_usage_summary(recorder: dict) -> dict:
    calls = recorder.get("llm_calls", [])
    prompt_values = [_safe_int(call.get("prompt_tokens", 0)) for call in calls]
    completion_values = [_safe_int(call.get("completion_tokens", 0)) for call in calls]
    reasoning_values = [_safe_int(call.get("reasoning_tokens", 0)) for call in calls]
    visible_values = [_safe_int(call.get("visible_output_tokens", 0)) for call in calls]
    return {
        "num_llm_calls": len(calls),
        "total_prompt_tokens": sum(prompt_values),
        "total_completion_tokens": sum(completion_values),
        "total_reasoning_tokens": sum(reasoning_values),
        "total_visible_output_tokens": sum(visible_values),
        "max_prompt_tokens_single_call": max(prompt_values, default=0),
        "max_completion_tokens_single_call": max(completion_values, default=0),
        "max_reasoning_tokens_single_call": max(reasoning_values, default=0),
    }


def _build_frame_summary(recorder: dict) -> dict:
    events = recorder.get("frame_calls", [])
    frame_values = [_safe_int(event.get("frames_sampled", 0)) for event in events]
    image_values = [_safe_int(event.get("image_inputs", 0)) for event in events]
    return {
        "num_tool_frame_calls": len(events),
        "total_frames_sampled": sum(frame_values),
        "total_image_inputs": sum(image_values),
        "max_frames_single_tool_call": max(frame_values, default=0),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run_artifacts(output_dir: Path, question: str, prediction: str, trajectory_payload: dict, recorder: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    usage_summary = _build_usage_summary(recorder)
    frame_summary = _build_frame_summary(recorder)
    _write_json(output_dir / "prediction.json", {"prediction": prediction})
    trajectory_payload = dict(trajectory_payload)
    trajectory_payload["llm_calls"] = recorder.get("llm_calls", [])
    trajectory_payload["tool_frame_calls"] = recorder.get("frame_calls", [])
    trajectory_payload["usage_summary"] = usage_summary
    trajectory_payload["frame_summary"] = frame_summary
    _write_json(output_dir / "trajectory.json", trajectory_payload)
    _write_json(
        output_dir / "metrics.json",
        {
            "usage_summary": usage_summary,
            "frame_summary": frame_summary,
            "llm_calls": recorder.get("llm_calls", []),
            "tool_frame_calls": recorder.get("frame_calls", []),
        },
    )
    return {"usage_summary": usage_summary, "frame_summary": frame_summary}


def _write_failure_artifacts(output_dir: Path, question: str, error: str, recorder: dict | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = recorder or {"llm_calls": [], "frame_calls": []}
    usage_summary = _build_usage_summary(recorder)
    frame_summary = _build_frame_summary(recorder)
    _write_json(output_dir / "prediction.json", {"prediction": "", "error": error})
    _write_json(
        output_dir / "trajectory.json",
        {
            "question": question,
            "steps": [],
            "total_steps": 0,
            "final_answer": "",
            "finish_reason": "error",
            "error": error,
            "llm_calls": recorder.get("llm_calls", []),
            "tool_frame_calls": recorder.get("frame_calls", []),
            "usage_summary": usage_summary,
            "frame_summary": frame_summary,
        },
    )
    _write_json(
        output_dir / "metrics.json",
        {
            "error": error,
            "usage_summary": usage_summary,
            "frame_summary": frame_summary,
            "llm_calls": recorder.get("llm_calls", []),
            "tool_frame_calls": recorder.get("frame_calls", []),
        },
    )
    return {"usage_summary": usage_summary, "frame_summary": frame_summary}


def _native_videoseek_sample_main(sample_request: dict, result_queue) -> None:
    recorder = {"llm_calls": [], "frame_calls": [], "video_path": sample_request["video_path"]}
    output_dir = Path(sample_request["output_dir"])
    question = sample_request["question"]

    from lmms_eval.models.videoseek_native.agent import VideoSeekAgent
    from lmms_eval.models.videoseek_native.config import build_agent_config
    from lmms_eval.models.videoseek_native.utils import extract_mcq_letter, set_call_label, set_thread_recorder

    set_thread_recorder(recorder)
    set_call_label(None)
    try:
        config = build_agent_config(sample_request["config"])
        agent = VideoSeekAgent(
            config=config,
            video_path=sample_request["video_path"],
            subtitle_path=sample_request["subtitle_path"],
            output_dir=str(output_dir),
            tools=config["tools"],
            verbose=sample_request["verbose"],
        )
        trajectory = agent.run(question)
        prediction = str(getattr(trajectory, "final_answer", "") or "")
        if sample_request["extract_answer"]:
            prediction = extract_mcq_letter(prediction)
        metrics = _write_run_artifacts(output_dir, question, prediction, trajectory.to_dict(), recorder)
        result_queue.put({"prediction": prediction, "metrics": metrics})
    except Exception as exc:
        error = str(exc).replace("\n", " ")[:500]
        metrics = _write_failure_artifacts(output_dir, question, error, recorder)
        result_queue.put({"prediction": "", "metrics": metrics, "error": error})
    finally:
        set_call_label(None)
        set_thread_recorder(None)


def _run_request_in_subprocess(sample_request: dict, timeout: int, target=_native_videoseek_sample_main) -> dict:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=target, args=(sample_request, result_queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError(f"VideoSeek sample timed out after {timeout} seconds")

    result = None
    try:
        result = result_queue.get(timeout=1)
    except queue.Empty:
        result = None
    result_queue.close()
    result_queue.cancel_join_thread()

    if result is None:
        raise RuntimeError(f"VideoSeek subprocess exited without a result (exit_code={process.exitcode})")
    return result


@register_model("videoseek")
class VideoSeek(lmms):
    """Native VideoSeek backend for lmms-eval with subprocess-isolated samples."""

    def __init__(
        self,
        model_name: str = "openai/qwen3.5-4b",
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str = "any",
        api_version: str = "",
        output_dir: str = "./logs/videoseek_runs",
        max_steps: int = 6,
        max_tokens: int = 4096,
        reasoning_effort: str = "medium",
        temperature: float = 0.0,
        seed: int = 42,
        verbose: bool = False,
        extract_answer: bool = True,
        timeout: int = 1800,
        num_workers: int = 1,
        sample_retry_attempts: int = 0,
        sample_retry_backoff_s: float = 2.0,
        run_name: str = "",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version
        self.output_root = Path(output_dir).expanduser().resolve()
        self.max_steps = int(max_steps)
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.verbose = _as_bool(verbose)
        self.extract_answer = _as_bool(extract_answer)
        self.timeout = int(timeout)
        self.num_workers = max(1, int(num_workers))
        self.sample_retry_attempts = max(0, int(sample_retry_attempts))
        self.sample_retry_backoff_s = max(0.0, float(sample_retry_backoff_s))
        run_suffix = run_name.strip() if isinstance(run_name, str) else ""
        if not run_suffix:
            run_suffix = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}"
        self.run_dir = self.output_root / f"run_{run_suffix}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_manifest()

    def _build_agent_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "api_version": self.api_version,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
        }

    def _write_run_manifest(self) -> None:
        _write_json(
            self.run_dir / "run_manifest.json",
            {
                "run_dir": str(self.run_dir),
                "model_name": self.model_name,
                "api_base": self.api_base,
                "max_steps": self.max_steps,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
                "temperature": self.temperature,
                "num_workers": self.num_workers,
                "sample_retry_attempts": self.sample_retry_attempts,
                "sample_retry_backoff_s": self.sample_retry_backoff_s,
                "timeout": self.timeout,
                "runtime_mode": "subprocess",
                "vendored_upstream_commit": VENDORED_UPSTREAM_COMMIT,
            },
        )

    def _write_run_summary(self, sample_results: List[dict]) -> None:
        aggregate = {
            "num_samples": len(sample_results),
            "total_prompt_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_prompt_tokens", 0)) for item in sample_results),
            "total_completion_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_completion_tokens", 0)) for item in sample_results),
            "total_reasoning_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_reasoning_tokens", 0)) for item in sample_results),
            "total_visible_output_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_visible_output_tokens", 0)) for item in sample_results),
            "max_prompt_tokens_single_call": max((_safe_int(item.get("usage_summary", {}).get("max_prompt_tokens_single_call", 0)) for item in sample_results), default=0),
            "total_frames_sampled": sum(_safe_int(item.get("frame_summary", {}).get("total_frames_sampled", 0)) for item in sample_results),
            "total_image_inputs": sum(_safe_int(item.get("frame_summary", {}).get("total_image_inputs", 0)) for item in sample_results),
        }
        _write_json(self.run_dir / "run_summary.json", {"run_dir": str(self.run_dir), "aggregate": aggregate, "samples": sample_results})

    def _run_sample(self, sample_request: dict) -> tuple[str, dict]:
        result = _run_request_in_subprocess(sample_request, timeout=self.timeout)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result.get("prediction", ""), result.get("metrics", {"usage_summary": {}, "frame_summary": {}})

    def generate_until(self, requests) -> List[str]:
        request_args = [reg.args for reg in requests]
        responses = [None] * len(request_args)
        sample_results: List[dict] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="VideoSeek Responding")

        def process_one(index: int, args) -> tuple[int, str, tuple, dict]:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = args
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)
            if not isinstance(visuals, list):
                visuals = [visuals]
            if not visuals or not isinstance(visuals[0], str):
                raise ValueError(f"VideoSeek expects a local video path, got: {visuals}")

            video_path = visuals[0]
            subtitle_path = _subtitle_path_for_video(video_path)
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task))
            output_dir = self.run_dir / f"{safe_task}_doc{doc_id}_idx{index}"
            response = ""
            sample_metrics = {"usage_summary": {}, "frame_summary": {}}
            last_error_msg = "empty prediction"
            total_attempts = 1 + self.sample_retry_attempts
            sample_request = {
                "video_path": video_path,
                "subtitle_path": subtitle_path,
                "question": context,
                "output_dir": str(output_dir),
                "config": self._build_agent_config(),
                "verbose": self.verbose,
                "extract_answer": self.extract_answer,
            }

            for attempt in range(1, total_attempts + 1):
                try:
                    prediction, sample_metrics = self._run_sample(sample_request)
                    if str(prediction).strip():
                        response = prediction
                        break
                    last_error_msg = "empty prediction"
                    raise ValueError(last_error_msg)
                except Exception as exc:
                    last_error_msg = str(exc).replace("\n", " ")[:500]
                    if attempt < total_attempts:
                        eval_logger.warning(
                            f"VideoSeek request failed for task={task} doc_id={doc_id} "
                            f"(attempt {attempt}/{total_attempts}): {last_error_msg}. Retrying."
                        )
                        if self.sample_retry_backoff_s > 0:
                            time.sleep(self.sample_retry_backoff_s * attempt)
                        continue
                    eval_logger.error(
                        f"VideoSeek request failed for task={task} doc_id={doc_id} "
                        f"after {total_attempts} attempts: {last_error_msg}"
                    )
                    sample_metrics = _write_failure_artifacts(output_dir, context, last_error_msg)

            sample_result = {
                "index": index,
                "task": str(task),
                "doc_id": int(doc_id),
                "sample_dir": str(output_dir),
                "prediction": response,
                "usage_summary": sample_metrics.get("usage_summary", {}),
                "frame_summary": sample_metrics.get("frame_summary", {}),
            }
            return index, response, (context, gen_kwargs), sample_result

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_one, index, args) for index, args in enumerate(request_args)]
            for future in as_completed(futures):
                index, response, cache_key, sample_result = future.result()
                responses[index] = response
                sample_results.append(sample_result)
                self.cache_hook.add_partial("generate_until", cache_key, response)
                pbar.update(1)

        pbar.close()
        self._write_run_summary(sorted(sample_results, key=lambda item: item["index"]))
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("VideoSeek only supports generate_until")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)
