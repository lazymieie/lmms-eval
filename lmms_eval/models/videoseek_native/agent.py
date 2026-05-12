import json
from abc import ABC, abstractmethod
from typing import List

from decord import VideoReader

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
        self.verbose = verbose

        self.duration = round(len(self.vr) / self.vr.get_avg_fps(), 2)
        self.subtitles = load_subtitles(subtitle_path)

        self.model_name = config["model_name"]
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.api_version = config["api_version"]
        self.max_steps = config["max_steps"]
        self.max_tokens = config["max_tokens"]
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

            with call_label("decision"):
                response = call_llm_api(
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
                        "content": f"Observation from `{str(logged_action.to_dict())}`:\n{outcome}",
                    }
                )

            if self.final_answer is not None:
                return Trajectory(question=question, steps=self.trajectory_steps, final_answer=self.final_answer, finish_reason="stop")

        self.messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of steps. "
                    f"Question:\n{question}\n\n"
                    "If the question is a multiple-choice question, please directly answer with the option's letter from the given choices without any additional text."
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
        return Trajectory(question=question, steps=self.trajectory_steps, final_answer=self.final_answer, finish_reason="reach_max_steps")
