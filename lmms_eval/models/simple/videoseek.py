import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from threading import Lock, local
from typing import Any, Dict, List, Tuple

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
_VIDEOSEEK_THREAD_STATE = local()


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _get_thread_recorder() -> dict | None:
    return getattr(_VIDEOSEEK_THREAD_STATE, "recorder", None)


def _set_thread_recorder(recorder: dict | None) -> None:
    _VIDEOSEEK_THREAD_STATE.recorder = recorder


def _get_thread_call_label() -> str | None:
    return getattr(_VIDEOSEEK_THREAD_STATE, "call_label", None)


def _set_thread_call_label(label: str | None) -> None:
    _VIDEOSEEK_THREAD_STATE.call_label = label


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
        run_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.videoseek_root = Path(videoseek_root).expanduser().resolve()
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version
        self.output_root = Path(output_dir).expanduser().resolve()
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
        run_suffix = run_name.strip() if isinstance(run_name, str) else ""
        if not run_suffix:
            run_suffix = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}"
        self.run_dir = self.output_root / f"run_{run_suffix}"
        self._warned_timeout = False
        self._warned_action_parse_mode = False

        if not self.videoseek_root.exists():
            raise FileNotFoundError(f"VideoSeek repo not found: {self.videoseek_root}")
        if not (self.videoseek_root / "videoseek" / "cli.py").exists():
            raise FileNotFoundError(f"VideoSeek CLI source not found under: {self.videoseek_root}")
        self._general_config, self._prompts_config, self._agent_cls = self._load_videoseek_components()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_manifest()

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
            utils_module = import_module("videoseek.utils")
            tools_package = import_module("videoseek.tools")
            overview_module = import_module("videoseek.tools.overview")
            skim_module = import_module("videoseek.tools.skim")
            focus_module = import_module("videoseek.tools.focus")
            answer_module = import_module("videoseek.tools.answer")
            self._install_videoseek_instrumentation(
                utils_module=utils_module,
                agent_module=agent_module,
                tools_package=tools_package,
                overview_module=overview_module,
                skim_module=skim_module,
                focus_module=focus_module,
                answer_module=answer_module,
            )
            return config_module.general_config, config_module.prompts_config, agent_module.VideoSeekAgent

    def _write_run_manifest(self) -> None:
        manifest = {
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
        }
        (self.run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _install_videoseek_instrumentation(
        self,
        *,
        utils_module,
        agent_module,
        tools_package,
        overview_module,
        skim_module,
        focus_module,
        answer_module,
    ) -> None:
        if getattr(utils_module, "_lmms_eval_instrumented", False):
            return

        original_call_llm_api = utils_module.call_llm_api

        def infer_call_type(messages, kwargs) -> str:
            explicit = _get_thread_call_label()
            if explicit:
                return explicit
            if kwargs.get("tool_choice") == "required" and kwargs.get("tools"):
                return "parse_actions"
            if messages:
                last_text = _message_text(messages[-1])
                if "You have reached the maximum number of steps." in last_text:
                    return "final_answer_fallback"
                if "Step [" in last_text and "Thinking Policy" in last_text:
                    return "thought"
            return "llm_call"

        def instrumented_call_llm_api(*args, **kwargs):
            recorder = _get_thread_recorder()
            messages = kwargs.get("messages") if "messages" in kwargs else (args[1] if len(args) > 1 else None)
            call_type = infer_call_type(messages or [], kwargs)
            started_at = time.time()
            response = None
            error = None
            try:
                response = original_call_llm_api(*args, **kwargs)
                return response
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                if recorder is None:
                    return
                usage = getattr(response, "usage", None) if response is not None else None
                prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", 0) if usage else 0)
                completion_tokens = _safe_int(getattr(usage, "completion_tokens", 0) if usage else 0)
                reasoning_tokens = 0
                if usage is not None:
                    completion_details = getattr(usage, "completion_tokens_details", None)
                    if completion_details is not None:
                        reasoning_tokens = _safe_int(getattr(completion_details, "reasoning_tokens", 0))
                    else:
                        reasoning_tokens = _safe_int(getattr(usage, "reasoning_tokens", 0))
                finish_reason = None
                content_chars = 0
                reasoning_chars = 0
                if response is not None and getattr(response, "choices", None):
                    choice = response.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    message = getattr(choice, "message", None)
                    if message is not None:
                        content_chars = len(str(getattr(message, "content", "") or ""))
                        reasoning_chars = len(str(getattr(message, "reasoning", "") or getattr(message, "reasoning_content", "") or ""))
                recorder["llm_calls"].append(
                    {
                        "call_index": len(recorder["llm_calls"]) + 1,
                        "call_type": call_type,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "visible_output_tokens": max(0, completion_tokens - reasoning_tokens) if completion_tokens else 0,
                        "finish_reason": finish_reason,
                        "content_chars": content_chars,
                        "reasoning_chars": reasoning_chars,
                        "elapsed_s": round(time.time() - started_at, 3),
                        "message_count": len(messages or []),
                        "error": error,
                    }
                )

        def make_tool_wrapper(tool_name: str, original_func):
            def wrapped_tool(*args, **kwargs):
                recorder = _get_thread_recorder()
                parameters = kwargs.get("parameters", {})
                event = {"tool_name": tool_name}
                if tool_name == "overview":
                    num_frames = self._general_config["frame_sampling_factor"] * self._general_config["overview_base"]
                    if num_frames % 8 != 0:
                        num_frames += 8 - (num_frames % 8)
                    event.update({"frames_sampled": int(num_frames), "image_inputs": int(num_frames // 8)})
                elif tool_name == "skim":
                    num_frames = self._general_config["frame_sampling_factor"] * self._general_config["skim_base"]
                    event.update(
                        {
                            "frames_sampled": int(num_frames),
                            "image_inputs": int(num_frames),
                            "start_time": parameters.get("start_time"),
                            "end_time": parameters.get("end_time"),
                        }
                    )
                elif tool_name == "focus":
                    vr = parameters.get("vr")
                    start_time = parameters.get("start_time")
                    end_time = parameters.get("end_time")
                    max_num_frames = self._general_config["frame_sampling_factor"] * self._general_config["focus_base"]
                    frames_sampled = 0
                    if vr is not None and start_time is not None and end_time is not None:
                        start_frame = int(float(start_time) * vr.get_avg_fps())
                        end_frame = min(int(float(end_time) * vr.get_avg_fps()), len(vr) - 1)
                        frames_sampled = max(0, min(int(end_frame - start_frame), max_num_frames))
                    event.update(
                        {
                            "frames_sampled": int(frames_sampled),
                            "image_inputs": int(frames_sampled),
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    )
                else:
                    event.update({"frames_sampled": 0, "image_inputs": 0})
                previous_label = _get_thread_call_label()
                _set_thread_call_label(tool_name)
                try:
                    return original_func(*args, **kwargs)
                finally:
                    _set_thread_call_label(previous_label)
                    if recorder is not None:
                        recorder["frame_calls"].append(event)

            return wrapped_tool

        original_exec_action = getattr(agent_module.VideoSeekAgent, "_VideoSeekAgent__exec_action")

        def tool_calls_to_actions(agent_self, tool_calls) -> list:
            actions = []
            answer_call_idx = None
            for tool_idx, tool_call in enumerate(tool_calls or []):
                if isinstance(tool_call, dict):
                    function_name = tool_call.get("function", {}).get("name")
                    arguments_text = tool_call.get("function", {}).get("arguments", "{}")
                    function_id = tool_call.get("id")
                else:
                    function_name = getattr(getattr(tool_call, "function", None), "name", None)
                    arguments_text = getattr(getattr(tool_call, "function", None), "arguments", "{}")
                    function_id = getattr(tool_call, "id", None)
                try:
                    parameters = json.loads(arguments_text or "{}")
                except Exception:
                    parameters = {}
                if agent_self.tool_registry.has_tool(function_name):
                    if function_name == "answer":
                        answer_call_idx = tool_idx
                    actions.append(agent_module.Action(function_name=function_name, parameters=parameters, function_id=function_id))
            if len(actions) > 1 and answer_call_idx is not None:
                actions.pop(answer_call_idx)
            return actions

        def patched_run(agent_self, question: str):
            agent_self.reset()
            agent_self.question = question
            subtitles_str = agent_module.convert_to_free_form_text_representation(agent_self.subtitles, content_type="subtitle")
            agent_self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Video Duration: {agent_self.duration:.01f}s\n\n"
                        f"Video Subtitles:\n{subtitles_str}\n\n"
                        f"Question:\n{question}"
                    ),
                }
            )

            for step in range(agent_self.max_steps):
                agent_self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Step [{step + 1} / {agent_self.max_steps}]: "
                            "Reason over the current state and directly choose the next tool call(s). "
                            "Follow the Tool Calling Policy and the Final Answer Policy. "
                            "Return tool calls only; do not provide extra narration outside the tool call response."
                        ),
                    }
                )

                previous_label = _get_thread_call_label()
                _set_thread_call_label("decision")
                try:
                    response = agent_module.call_llm_api(
                        messages=agent_self.messages,
                        model_name=agent_self.model_name,
                        api_base=agent_self.api_base,
                        api_key=agent_self.api_key,
                        api_version=agent_self.api_version,
                        max_tokens=agent_self.max_tokens,
                        reasoning_effort=agent_self.reasoning_effort,
                        seed=agent_self.seed,
                        tools=agent_self.tools,
                        tool_choice="required",
                        temperature=agent_self.temperature,
                    )
                finally:
                    _set_thread_call_label(previous_label)

                message = response.choices[0].message if response is not None and getattr(response, "choices", None) else None
                visible_content = str(getattr(message, "content", "") or "")
                reasoning_text = str(getattr(message, "reasoning", "") or getattr(message, "reasoning_content", "") or "")
                thought = reasoning_text or visible_content or ""
                actions = tool_calls_to_actions(agent_self, getattr(message, "tool_calls", None) if message is not None else None)

                assistant_message = {"role": "assistant", "content": visible_content}
                if actions and actions[0].function_name != "answer":
                    assistant_message["tool_calls"] = [
                        {
                            "id": action.function_id,
                            "type": "function",
                            "function": {
                                "name": action.function_name,
                                "arguments": str(action.parameters),
                            },
                        }
                        for action in actions
                    ]
                agent_self.messages.append(assistant_message)

                for action in actions:
                    try:
                        outcome = original_exec_action(agent_self, action)
                    except Exception:
                        outcome = "Tool execution failed."
                    observation = agent_module.Observation(action=action, outcome=outcome)
                    if action.parameters is not None:
                        action.parameters.pop("vr", None)
                        action.parameters.pop("subtitles", None)
                    agent_self.trajectory_steps.append(
                        agent_module.TrajectoryStep(
                            step_id=step + 1,
                            thought=thought,
                            action=action,
                            observation=observation,
                        )
                    )
                    if action.function_name == "answer":
                        agent_self.final_answer = observation.outcome
                        break
                    agent_self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": action.function_id,
                            "content": f"Observation from `{str(action.to_dict())}`:\n{outcome}",
                        }
                    )

                if len(actions) == 0:
                    agent_self.messages.append(
                        {
                            "role": "user",
                            "content": "There is no valid function call in your response. You must emit a valid tool call in each response.",
                        }
                    )
                    continue

                if agent_self.final_answer is not None:
                    break

            if agent_self.final_answer is None:
                agent_self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of steps. "
                            f"Question:\n{question}\n\n"
                            "If the question is a multiple-choice question, please directly answer with the option's letter from the given choices without any additional text."
                        ),
                    }
                )
                previous_label = _get_thread_call_label()
                _set_thread_call_label("final_answer_fallback")
                try:
                    response = agent_module.call_llm_api(
                        messages=agent_self.messages,
                        model_name=agent_self.model_name,
                        api_base=agent_self.api_base,
                        api_key=agent_self.api_key,
                        api_version=agent_self.api_version,
                        max_tokens=agent_self.max_tokens,
                        reasoning_effort=agent_self.reasoning_effort,
                        seed=agent_self.seed,
                        temperature=agent_self.temperature,
                    )
                finally:
                    _set_thread_call_label(previous_label)
                agent_self.final_answer = response.choices[0].message.content
                agent_self.messages.append({"role": "assistant", "content": agent_self.final_answer})
                return agent_module.Trajectory(
                    question=question,
                    steps=agent_self.trajectory_steps,
                    final_answer=agent_self.final_answer,
                    finish_reason="reach_max_steps",
                )

            return agent_module.Trajectory(
                question=question,
                steps=agent_self.trajectory_steps,
                final_answer=agent_self.final_answer,
                finish_reason="stop",
            )

        instrumented_overview = make_tool_wrapper("overview", overview_module.execute_overview)
        instrumented_skim = make_tool_wrapper("skim", skim_module.execute_skim)
        instrumented_focus = make_tool_wrapper("focus", focus_module.execute_focus)
        instrumented_answer = make_tool_wrapper("answer", answer_module.execute_answer)

        utils_module.call_llm_api = instrumented_call_llm_api
        agent_module.call_llm_api = instrumented_call_llm_api
        overview_module.call_llm_api = instrumented_call_llm_api
        skim_module.call_llm_api = instrumented_call_llm_api
        focus_module.call_llm_api = instrumented_call_llm_api
        answer_module.call_llm_api = instrumented_call_llm_api
        overview_module.execute_overview = instrumented_overview
        skim_module.execute_skim = instrumented_skim
        focus_module.execute_focus = instrumented_focus
        answer_module.execute_answer = instrumented_answer
        tools_package.TOOL_FUNCTIONS["overview"] = instrumented_overview
        tools_package.TOOL_FUNCTIONS["skim"] = instrumented_skim
        tools_package.TOOL_FUNCTIONS["focus"] = instrumented_focus
        tools_package.TOOL_FUNCTIONS["answer"] = instrumented_answer
        tools_package.DEFAULT_TOOL_REGISTRY._tool_functions["overview"] = instrumented_overview
        tools_package.DEFAULT_TOOL_REGISTRY._tool_functions["skim"] = instrumented_skim
        tools_package.DEFAULT_TOOL_REGISTRY._tool_functions["focus"] = instrumented_focus
        tools_package.DEFAULT_TOOL_REGISTRY._tool_functions["answer"] = instrumented_answer
        agent_module.VideoSeekAgent.run = patched_run
        utils_module._lmms_eval_instrumented = True

    def _build_usage_summary(self, recorder: dict) -> dict:
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

    def _build_frame_summary(self, recorder: dict) -> dict:
        events = recorder.get("frame_calls", [])
        frame_values = [_safe_int(event.get("frames_sampled", 0)) for event in events]
        image_values = [_safe_int(event.get("image_inputs", 0)) for event in events]
        return {
            "num_tool_frame_calls": len(events),
            "total_frames_sampled": sum(frame_values),
            "total_image_inputs": sum(image_values),
            "max_frames_single_tool_call": max(frame_values, default=0),
        }

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

    def _write_run_artifacts(self, output_dir: Path, question: str, prediction: str, trajectory, recorder: dict) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        usage_summary = self._build_usage_summary(recorder)
        frame_summary = self._build_frame_summary(recorder)
        (output_dir / "prediction.json").write_text(
            json.dumps({"prediction": prediction}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        trajectory_payload = trajectory.to_dict() if hasattr(trajectory, "to_dict") else {"question": question, "final_answer": prediction}
        trajectory_payload["llm_calls"] = recorder.get("llm_calls", [])
        trajectory_payload["tool_frame_calls"] = recorder.get("frame_calls", [])
        trajectory_payload["usage_summary"] = usage_summary
        trajectory_payload["frame_summary"] = frame_summary
        (output_dir / "trajectory.json").write_text(
            json.dumps(trajectory_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "usage_summary": usage_summary,
                    "frame_summary": frame_summary,
                    "llm_calls": recorder.get("llm_calls", []),
                    "tool_frame_calls": recorder.get("frame_calls", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"usage_summary": usage_summary, "frame_summary": frame_summary}

    def _write_failure_artifacts(self, output_dir: Path, question: str, error: str, recorder: dict | None = None) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        recorder = recorder or {"llm_calls": [], "frame_calls": []}
        usage_summary = self._build_usage_summary(recorder)
        frame_summary = self._build_frame_summary(recorder)
        (output_dir / "prediction.json").write_text(
            json.dumps({"prediction": "", "error": error}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "question": question,
                    "steps": [],
                    "final_answer": "",
                    "finish_reason": "error",
                    "error": error,
                    "llm_calls": recorder.get("llm_calls", []),
                    "tool_frame_calls": recorder.get("frame_calls", []),
                    "usage_summary": usage_summary,
                    "frame_summary": frame_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "error": error,
                    "usage_summary": usage_summary,
                    "frame_summary": frame_summary,
                    "llm_calls": recorder.get("llm_calls", []),
                    "tool_frame_calls": recorder.get("frame_calls", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"usage_summary": usage_summary, "frame_summary": frame_summary}

    def _run_videoseek(self, video_path: str, subtitle_path: str | None, question: str, output_dir: Path) -> tuple[str, dict]:
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

        recorder = {"llm_calls": [], "frame_calls": [], "video_path": video_path}
        _set_thread_recorder(recorder)
        _set_thread_call_label(None)
        try:
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
            sample_metrics = self._write_run_artifacts(output_dir, question, prediction, trajectory, recorder)
            return prediction, sample_metrics
        finally:
            _set_thread_call_label(None)
            _set_thread_recorder(None)

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
        payload = {"run_dir": str(self.run_dir), "aggregate": aggregate, "samples": sample_results}
        (self.run_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
            subtitle_path = self._subtitle_path_for_video(video_path)
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task))
            output_dir = self.run_dir / f"{safe_task}_doc{doc_id}_idx{index}"
            response = ""
            sample_metrics = {"usage_summary": {}, "frame_summary": {}}
            last_error_msg = "empty prediction"
            total_attempts = 1 + self.sample_retry_attempts
            for attempt in range(1, total_attempts + 1):
                try:
                    prediction, sample_metrics = self._run_videoseek(video_path, subtitle_path, context, output_dir)
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
                    sample_metrics = self._write_failure_artifacts(output_dir, context, last_error_msg)
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
