import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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


@register_model("videoseek")
class VideoSeek(lmms):
    """VideoSeek CLI adapter for lmms-eval.

    This wrapper intentionally shells out to ``videoseek-cli`` so the first
    integration follows the upstream package behavior closely.
    """

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

        if not self.videoseek_root.exists():
            raise FileNotFoundError(f"VideoSeek repo not found: {self.videoseek_root}")
        if not (self.videoseek_root / "videoseek" / "cli.py").exists():
            raise FileNotFoundError(f"VideoSeek CLI source not found under: {self.videoseek_root}")

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

    def _latest_prediction(self, output_dir: Path, before: float) -> str:
        prediction_files = [
            path
            for path in output_dir.glob("*/prediction.json")
            if path.stat().st_mtime >= before
        ]
        if not prediction_files:
            raise FileNotFoundError(f"No VideoSeek prediction.json found in {output_dir}")
        latest = max(prediction_files, key=lambda path: path.stat().st_mtime)
        payload = json.loads(latest.read_text(encoding="utf-8"))
        return str(payload.get("prediction", ""))

    def _run_videoseek(self, video_path: str, subtitle_path: str | None, question: str, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        cli_bootstrap = "import litellm; litellm.drop_params=True; from videoseek.cli import main; raise SystemExit(main())"
        cmd = [
            sys.executable,
            "-c",
            cli_bootstrap,
            "--video_path",
            video_path,
            "--user_query",
            question,
            "--output_dir",
            str(output_dir),
            "--model_name",
            self.model_name,
            "--api_base",
            "OPENAI_API_BASE",
            "--api_key",
            "OPENAI_API_KEY",
            "--seed",
            str(self.seed),
            "--temperature",
            str(self.temperature),
            "--max_tokens",
            str(self.max_tokens),
            "--max_steps",
            str(self.max_steps),
            "--action_parse_mode",
            self.action_parse_mode,
        ]
        if self.reasoning_effort.lower() not in {"", "none", "null", "off", "false"}:
            cmd.extend(["--reasoning_effort", self.reasoning_effort])
        if self.api_version:
            cmd.extend(["--api_version", self.api_version])
        if subtitle_path:
            cmd.extend(["--subtitle_path", subtitle_path])
        if self.verbose:
            cmd.append("--verbose")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.videoseek_root}:{env.get('PYTHONPATH', '')}"
        env["OPENAI_API_BASE"] = self.api_base
        env["OPENAI_API_KEY"] = self.api_key
        started_at = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(self.videoseek_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "VideoSeek failed with exit code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        if result.stderr.strip():
            eval_logger.debug(f"VideoSeek stderr: {result.stderr.strip()}")
        return self._latest_prediction(output_dir, started_at)

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
            prediction = self._run_videoseek(video_path, subtitle_path, context, output_dir)
            response = _extract_mcq_letter(prediction) if self.extract_answer else prediction
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
