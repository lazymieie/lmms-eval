import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import List, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


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


def _extract_mcq_letter(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else text


_VIDEOSEEK_IMPORT_LOCK = Lock()


@register_model("videoseek")
class VideoSeek(lmms):
    """In-process VideoSeek adapter for lmms-eval."""

    def __init__(
        self,
        videoseek_root: str = "/gemini/space/gjx/videoseek",
        model_name: str = "openai/qwen3.5-4b",
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str = "any",
        api_version: str = "",
        output_dir: str = "./logs/videoseek_runs",
        max_steps: int = 20,
        max_tokens: int = 32768,
        reasoning_effort: str = "medium",
        action_parse_mode: str = "tool_call",
        temperature: float = 1.0,
        seed: int = 42,
        verbose: bool = False,
        extract_answer: bool = True,
        timeout: int = 1800,
        num_workers: int = 1,
        sample_retry_attempts: int = 2,
        sample_retry_backoff_s: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.videoseek_root = Path(videoseek_root).expanduser().resolve()
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.max_steps = int(max_steps)
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        self.action_parse_mode = action_parse_mode
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.verbose = _as_bool(verbose)
        self.extract_answer = _as_bool(extract_answer)
        self.timeout = int(timeout)
        self.num_workers = max(1, int(num_workers))
        self.sample_retry_attempts = max(0, int(sample_retry_attempts))
        self.sample_retry_backoff_s = max(0.0, float(sample_retry_backoff_s))
        self._warned_timeout = False
        self._warned_action_parse_mode = False

        if not self.videoseek_root.exists():
            raise FileNotFoundError(f"VideoSeek repo not found: {self.videoseek_root}")
        if not (self.videoseek_root / "videoseek" / "cli.py").exists():
            raise FileNotFoundError(f"VideoSeek CLI source not found under: {self.videoseek_root}")
        self._general_config, self._prompts_config, self._agent_cls = self._load_videoseek_components()

    def _subtitle_path_for_video(self, video_path: str) -> str | None:
        path = Path(video_path)
        candidates = [
            path.with_suffix(".srt"),
            path.parent.parent / "subtitle" / f"{path.stem}.srt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _load_videoseek_components(self):
        with _VIDEOSEEK_IMPORT_LOCK:
            root = str(self.videoseek_root)
            if root not in sys.path:
                sys.path.insert(0, root)

            config_module = sys.modules.get("config")
            config_path = str((self.videoseek_root / "config" / "__init__.py").resolve())
            if config_module is None or Path(getattr(config_module, "__file__", "")).resolve() != Path(config_path):
                sys.modules.pop("config", None)
                config_module = import_module("config")

            agent_module = import_module("videoseek.agent")
            return config_module.general_config, config_module.prompts_config, agent_module.VideoSeekAgent

    def _build_agent_config(self) -> dict:
        config = dict(self._general_config)
        config.update(self._prompts_config)
        config["model_name"] = self.model_name
        config["api_base"] = self.api_base
        config["api_key"] = self.api_key
        config["api_version"] = self.api_version
        config["reasoning_effort"] = self.reasoning_effort
        config["seed"] = self.seed
        config["temperature"] = self.temperature
        config["max_tokens"] = self.max_tokens
        config["max_steps"] = self.max_steps
        return config

    def _write_run_artifacts(self, output_dir: Path, question: str, prediction: str, trajectory) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / f"run_{time.time_ns()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "prediction.json").write_text(
            json.dumps({"prediction": prediction}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        trajectory_payload = trajectory.to_dict() if hasattr(trajectory, "to_dict") else {"question": question, "final_answer": prediction}
        (run_dir / "trajectory.json").write_text(
            json.dumps(trajectory_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_failure_artifacts(self, output_dir: Path, question: str, error: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / f"failed_{time.time_ns()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "prediction.json").write_text(
            json.dumps({"prediction": "", "error": error}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "question": question,
                    "steps": [],
                    "final_answer": "",
                    "finish_reason": "error",
                    "error": error,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _run_videoseek(self, video_path: str, subtitle_path: str | None, question: str, output_dir: Path) -> str:
        if self.timeout and not self._warned_timeout:
            eval_logger.warning(
                "VideoSeek now runs in-process. `timeout` is soft-only; Python cannot forcibly stop a hung agent thread."
            )
            self._warned_timeout = True
        if self.action_parse_mode and self.action_parse_mode != "tool_call" and not self._warned_action_parse_mode:
            eval_logger.warning(
                f"VideoSeek in-process adapter ignores action_parse_mode={self.action_parse_mode!r}; upstream agent uses tool-call parsing."
            )
            self._warned_action_parse_mode = True

        agent = self._agent_cls(
            config=self._build_agent_config(),
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_dir=str(output_dir),
            tools=self._general_config["tools"],
            verbose=self.verbose,
        )
        trajectory = agent.run(question)
        prediction = str(getattr(trajectory, "final_answer", "") or "")
        self._write_run_artifacts(output_dir, question, prediction, trajectory)
        return prediction

    def generate_until(self, requests) -> List[str]:
        request_args = [reg.args for reg in requests]
        responses = [None] * len(request_args)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="VideoSeek Responding")

        def process_one(index: int, args) -> tuple[int, str, tuple]:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = args
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)
            if not isinstance(visuals, list):
                visuals = [visuals]
            if not visuals or not isinstance(visuals[0], str):
                raise ValueError(f"VideoSeek expects a local video path, got: {visuals}")

            video_path = visuals[0]
            subtitle_path = self._subtitle_path_for_video(video_path)
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task))
            output_dir = self.output_dir / f"{safe_task}_doc{doc_id}_idx{index}"
            response = ""
            last_error_msg = "empty prediction"
            total_attempts = 1 + self.sample_retry_attempts
            for attempt in range(1, total_attempts + 1):
                try:
                    prediction = self._run_videoseek(video_path, subtitle_path, context, output_dir)
                    candidate = _extract_mcq_letter(prediction) if self.extract_answer else prediction
                    if str(candidate).strip():
                        response = candidate
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
                    self._write_failure_artifacts(output_dir, context, last_error_msg)
            return index, response, (context, gen_kwargs)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_one, index, args) for index, args in enumerate(request_args)]
            for future in as_completed(futures):
                index, response, cache_key = future.result()
                responses[index] = response
                self.cache_hook.add_partial("generate_until", cache_key, response)
                pbar.update(1)

        pbar.close()
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("VideoSeek only supports generate_until")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)
