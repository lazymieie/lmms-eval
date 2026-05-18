import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from decord import VideoReader
from loguru import logger as eval_logger

from .core import Action, Observation, Trajectory, TrajectoryStep
from .tools import DEFAULT_TOOL_REGISTRY
from .utils import call_label, call_llm_api, convert_to_free_form_text_representation, load_subtitles


class BaseAgent(ABC):
    def __init__(self) -> None:
        self.messages: List[dict] = []
        self.final_answer = None
        self.question = None

    def reset(self) -> None:
        self.messages = self.construct_initial_messages()
        self.final_answer = None
        self.question = None

    @abstractmethod
    def construct_initial_messages(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str) -> Trajectory:
        raise NotImplementedError


class VideoSeekAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        video_path: str,
        subtitle_path: str | None,
        output_dir: str,
        tools: list,
        verbose: bool = False,
    ):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.vr = VideoReader(video_path)
        self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.tools = self.tool_registry.resolve_tools(tools + ["answer"])
        self.output_dir = output_dir
        self.output_dir_path = Path(output_dir)
        self.verbose = verbose

        self.duration = round(len(self.vr) / self.vr.get_avg_fps(), 2)
        self.subtitles = load_subtitles(subtitle_path)

        self.model_name = config["model_name"]
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.api_version = config["api_version"]
        self.max_steps = config["max_steps"]
        self.max_tokens = config["max_tokens"]
        self.observation_max_chars = max(256, int(config.get("observation_max_chars", 1600)))
        self.decision_retry_attempts = max(1, int(config.get("decision_retry_attempts", 3)))
        self.decision_retry_backoff_s = max(0.0, float(config.get("decision_retry_backoff_s", 1.0)))
        self.reasoning_effort = config["reasoning_effort"]
        self.seed = config["seed"]
        self.temperature = config["temperature"]
        self.timeout = config.get("timeout", 900)

        self.messages = self.construct_initial_messages()
        self.trajectory_steps: List[TrajectoryStep] = []

    def reset(self):
        super().reset()
        self.trajectory_steps = []

    def construct_initial_messages(self) -> List[dict]:
        system_prompt = self.config["SYSTEM_PROMPT"].format(
            overview_num_frames=self.config["frame_sampling_factor"] * self.config["overview_base"],
            skim_num_frames=self.config["frame_sampling_factor"] * self.config["skim_base"],
            focus_num_frames=self.config["frame_sampling_factor"] * self.config["focus_base"],
        )
        return [{"role": "system", "content": system_prompt}]

    def _parse_actions(self, tool_calls) -> List[Action]:
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
            except Exception as exc:
                raise ValueError(f"Invalid tool call arguments for {function_name}: {exc}") from exc

            if self.tool_registry.has_tool(function_name):
                if function_name == "answer":
                    answer_call_idx = tool_idx
                actions.append(Action(function_name=function_name, parameters=parameters, function_id=function_id))

        if len(actions) > 1 and answer_call_idx is not None:
            actions.pop(answer_call_idx)
        return actions

    def _exec_action(self, action: Action) -> str:
        function_name = getattr(action, "function_name", None) if action else None
        parameters = getattr(action, "parameters", {}) if action else {}

        if function_name == "answer":
            parameters = {"question": self.question, "messages": list(self.messages)}
        else:
            parameters = dict(parameters)
            parameters.update({"vr": self.vr, "subtitles": self.subtitles})

        if self.tool_registry.has_tool(function_name):
            outcome = self.tool_registry.get_function(function_name)(config=self.config, parameters=parameters)
            if outcome is None:
                return "Tool execution failed."
            return outcome
        raise ValueError(f"Invalid function name: {function_name}")

    def _compact_observation_for_history(self, action: Action, outcome: str) -> str:
        text = str(outcome or "").strip()
        if not text:
            return "No observation returned."
        if len(text) <= self.observation_max_chars:
            return text
        head_budget = max(256, self.observation_max_chars // 2)
        tail_budget = max(128, self.observation_max_chars - head_budget - 64)
        compacted = (
            text[:head_budget].rstrip()
            + "\n\n[Observation truncated for history to control context length]\n\n"
            + text[-tail_budget:].lstrip()
        )
        return compacted

    @staticmethod
    def _is_tool_call_json_error(exc: Exception) -> bool:
        error_text = str(exc)
        return (
            "Invalid JSON" in error_text
            and "EOF while parsing a list" in error_text
            or "json_invalid" in error_text
            or "tool_calls" in error_text and "validation error" in error_text
        )

    @staticmethod
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

    @staticmethod
    def _extract_json_error_location(error_text: str) -> dict:
        match = re.search(r"line (\d+) column (\d+)", error_text)
        if not match:
            return {"line": None, "column": None}
        return {"line": int(match.group(1)), "column": int(match.group(2))}

    def _infer_json_error_cause(self, error_text: str, error_location: dict) -> str:
        line_no = error_location.get("line")
        if "EOF while parsing a list" in error_text:
            if line_no is not None and line_no >= 5000:
                return "likely_truncated_or_overlong_tool_call_output"
            return "likely_incomplete_tool_call_json"
        if "json_invalid" in error_text or "validation error" in error_text:
            return "likely_malformed_tool_call_json"
        return "unknown_tool_call_json_error"

    def _write_decision_error_debug(self, step: int, attempt: int, exc: Exception) -> None:
        error_text = str(exc)
        error_location = self._extract_json_error_location(error_text)
        message_texts = [self._message_text(message) for message in self.messages]
        total_text_chars = sum(len(text) for text in message_texts)
        last_message_text = message_texts[-1] if message_texts else ""
        snapshot = {
            "step": step + 1,
            "attempt": attempt,
            "error_type": type(exc).__name__,
            "error": error_text,
            "error_location": error_location,
            "likely_cause": self._infer_json_error_cause(error_text, error_location),
            "message_count": len(self.messages),
            "trajectory_steps_count": len(self.trajectory_steps),
            "subtitle_count": len(self.subtitles),
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "total_message_text_chars": total_text_chars,
            "last_message_text_chars": len(last_message_text),
            "last_message_preview": last_message_text[:1000],
            "messages": self.messages,
        }
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        debug_path = self.output_dir_path / "decision_error_debug.jsonl"
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        detail_path = self.output_dir_path / f"decision_error_step{step + 1}_attempt{attempt}.json"
        detail_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        eval_logger.warning(
            "VideoSeek decision JSON error: "
            f"step={step + 1} attempt={attempt}/{self.decision_retry_attempts} "
            f"line={error_location.get('line')} column={error_location.get('column')} "
            f"messages={len(self.messages)} text_chars={total_text_chars} "
            f"likely_cause={snapshot['likely_cause']} "
            f"debug_file={detail_path}"
        )

    def _call_final_answer(self, question: str, finish_reason: str) -> Trajectory:
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the final answer stage. "
                    f"Question:\n{question}\n\n"
                    "If the question is a multiple-choice question, directly answer with only the option letter from the given choices."
                ),
            }
        )
        with call_label("final_answer_fallback"):
            response = call_llm_api(
                messages=self.messages,
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        self.final_answer = str(response.choices[0].message.content or "")
        self.messages.append({"role": "assistant", "content": self.final_answer})
        return Trajectory(question=question, steps=self.trajectory_steps, final_answer=self.final_answer, finish_reason=finish_reason)

    def _call_decision(self, step: int):
        last_error = None
        for attempt in range(1, self.decision_retry_attempts + 1):
            try:
                with call_label("decision"):
                    return call_llm_api(
                        messages=self.messages,
                        model_name=self.model_name,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        api_version=self.api_version,
                        max_tokens=self.max_tokens,
                        reasoning_effort=self.reasoning_effort,
                        seed=self.seed,
                        tools=self.tools,
                        tool_choice="required",
                        temperature=self.temperature,
                        timeout=self.timeout,
                    )
            except Exception as exc:
                last_error = exc
                if self._is_tool_call_json_error(exc):
                    self._write_decision_error_debug(step, attempt, exc)
                if not self._is_tool_call_json_error(exc) or attempt >= self.decision_retry_attempts:
                    break
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous tool-calling response for step {step + 1} was malformed. "
                            "Return exactly one valid tool call. Keep arguments minimal and do not include extra narration."
                        ),
                    }
                )
                if self.decision_retry_backoff_s > 0:
                    time.sleep(self.decision_retry_backoff_s * attempt)
        raise last_error

    def run(self, question: str) -> Trajectory:
        self.reset()
        self.question = question
        subtitles_str = convert_to_free_form_text_representation(self.subtitles, content_type="subtitle")

        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Video Duration: {self.duration:.01f}s\n\n"
                    f"Video Subtitles:\n{subtitles_str}\n\n"
                    f"Question:\n{question}"
                ),
            }
        )

        for step in range(self.max_steps):
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Step [{step + 1} / {self.max_steps}]: "
                        "Reason over the current state and directly choose the next tool call(s). "
                        "Follow the Tool Calling Policy and the Final Answer Policy. "
                        "Return tool calls only; do not provide extra narration outside the tool call response."
                    ),
                }
            )

            try:
                response = self._call_decision(step)
            except Exception as exc:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool-calling failed repeatedly. "
                            "Use the evidence already collected and provide the final answer directly."
                        ),
                    }
                )
                return self._call_final_answer(question, finish_reason=f"decision_error_fallback: {exc}")

            message = response.choices[0].message if response is not None and getattr(response, "choices", None) else None
            visible_content = str(getattr(message, "content", "") or "")
            reasoning_text = str(getattr(message, "reasoning", "") or getattr(message, "reasoning_content", "") or "")
            thought = reasoning_text or visible_content or ""
            assistant_message = {"role": "assistant", "content": visible_content}

            try:
                actions = self._parse_actions(getattr(message, "tool_calls", None) if message is not None else None)
            except Exception:
                actions = []

            if actions and actions[0].function_name != "answer":
                assistant_message["tool_calls"] = [
                    {
                        "id": action.function_id,
                        "type": "function",
                        "function": {"name": action.function_name, "arguments": json.dumps(action.parameters, ensure_ascii=False)},
                    }
                    for action in actions
                ]
            self.messages.append(assistant_message)

            if len(actions) == 0:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "There is no valid function call in your response. You must emit a valid tool call in each response.",
                    }
                )
                continue

            for action in actions:
                try:
                    outcome = self._exec_action(action)
                except Exception as exc:
                    outcome = f"Tool execution failed. Error: {exc}"

                logged_action = Action(function_name=action.function_name, parameters=dict(action.parameters or {}), function_id=action.function_id)
                observation = Observation(action=logged_action, outcome=outcome)
                self.trajectory_steps.append(
                    TrajectoryStep(step_id=step + 1, thought=thought, action=logged_action, observation=observation)
                )
                if action.function_name == "answer":
                    self.final_answer = outcome
                    break
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.function_id or action.function_name,
                        "content": f"Observation from `{str(logged_action.to_dict())}`:\n{self._compact_observation_for_history(logged_action, outcome)}",
                    }
                )

            if self.final_answer is not None:
                return Trajectory(question=question, steps=self.trajectory_steps, final_answer=self.final_answer, finish_reason="stop")

        return self._call_final_answer(question, finish_reason="reach_max_steps")
