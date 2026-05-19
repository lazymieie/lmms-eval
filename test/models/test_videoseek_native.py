import importlib
import json
import sys
import time
import types
from pathlib import Path

import pytest

from lmms_eval.models.simple import videoseek as videoseek_module


def _sleep_worker(_sample_request, _result_queue):
    time.sleep(2)


def _make_request(context: str, video_path: str, doc_id: int):
    doc_to_visual = lambda doc: [doc["video_path"]]
    return types.SimpleNamespace(args=(context, {}, doc_to_visual, doc_id, "videomme", "validation"))


def test_videoseek_rejects_legacy_model_args():
    with pytest.raises(TypeError):
        videoseek_module.VideoSeek.create_from_arg_string("videoseek_root=/tmp")

    with pytest.raises(TypeError):
        videoseek_module.VideoSeek.create_from_arg_string("action_parse_mode=tool_call")


def test_videoseek_initializes_without_external_repo(tmp_path):
    model = videoseek_module.VideoSeek(output_dir=str(tmp_path), run_name="native")
    manifest = json.loads((model.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_mode"] == "subprocess"
    assert manifest["vendored_upstream_commit"] == videoseek_module.VENDORED_UPSTREAM_COMMIT


def test_run_request_in_subprocess_times_out():
    with pytest.raises(TimeoutError):
        videoseek_module._run_request_in_subprocess({}, timeout=1, target=_sleep_worker)


def test_generate_until_preserves_order_and_writes_summary(monkeypatch, tmp_path):
    def fake_run_request(sample_request, timeout, target=videoseek_module._native_videoseek_sample_main):
        suffix = sample_request["question"].split()[-1]
        return {
            "prediction": suffix,
            "metrics": {
                "usage_summary": {
                    "total_prompt_tokens": 1,
                    "total_completion_tokens": 2,
                    "total_reasoning_tokens": 0,
                    "total_visible_output_tokens": 2,
                    "max_prompt_tokens_single_call": 1,
                },
                "frame_summary": {
                    "total_frames_sampled": 3,
                    "total_image_inputs": 4,
                },
            },
        }

    monkeypatch.setattr(videoseek_module, "_run_request_in_subprocess", fake_run_request)

    model = videoseek_module.VideoSeek(output_dir=str(tmp_path), run_name="ordered", num_workers=2)
    model.task_dict = {
        "videomme": {
            "validation": {
                0: {"video_path": "/tmp/video0.mp4"},
                1: {"video_path": "/tmp/video1.mp4"},
            }
        }
    }
    requests = [_make_request("question 0", "/tmp/video0.mp4", 0), _make_request("question 1", "/tmp/video1.mp4", 1)]

    outputs = model.generate_until(requests)

    assert outputs == ["0", "1"]
    summary = json.loads((model.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["num_samples"] == 2
    assert [sample["prediction"] for sample in summary["samples"]] == ["0", "1"]


def test_videoseek_backend_does_not_manage_subtitles(monkeypatch, tmp_path):
    captured = []

    def fake_run_request(sample_request, timeout, target=videoseek_module._native_videoseek_sample_main):
        captured.append(sample_request)
        return {
            "prediction": "A",
            "metrics": {
                "usage_summary": {
                    "total_prompt_tokens": 1,
                    "total_completion_tokens": 1,
                    "total_reasoning_tokens": 0,
                    "total_visible_output_tokens": 1,
                    "max_prompt_tokens_single_call": 1,
                },
                "frame_summary": {
                    "total_frames_sampled": 1,
                    "total_image_inputs": 1,
                },
            },
        }

    monkeypatch.setattr(videoseek_module, "_run_request_in_subprocess", fake_run_request)
    model = videoseek_module.VideoSeek(output_dir=str(tmp_path), run_name="subtitle-gating", num_workers=1)
    model.task_dict = {
        "videomme_long": {"validation": {0: {"video_path": "/tmp/video0.mp4"}}},
        "videomme_long_w_subtitle": {"validation": {1: {"video_path": "/tmp/video1.mp4"}}},
    }

    req_no_sub = types.SimpleNamespace(args=("question 0", {}, lambda doc: [doc["video_path"]], 0, "videomme_long", "validation"))
    req_with_sub = types.SimpleNamespace(args=("question 1", {}, lambda doc: [doc["video_path"]], 1, "videomme_long_w_subtitle", "validation"))

    outputs = model.generate_until([req_no_sub, req_with_sub])

    assert outputs == ["A", "A"]
    assert "subtitle_path" not in captured[0]
    assert "subtitle_path" not in captured[1]


def test_timeout_failure_writes_artifacts(monkeypatch, tmp_path):
    def timeout_request(_sample_request, timeout, target=videoseek_module._native_videoseek_sample_main):
        raise TimeoutError(f"VideoSeek sample timed out after {timeout} seconds")

    monkeypatch.setattr(videoseek_module, "_run_request_in_subprocess", timeout_request)

    model = videoseek_module.VideoSeek(output_dir=str(tmp_path), run_name="timeout", num_workers=1)
    model.task_dict = {
        "videomme": {
            "validation": {
                0: {"video_path": "/tmp/video0.mp4"},
            }
        }
    }
    outputs = model.generate_until([_make_request("question 0", "/tmp/video0.mp4", 0)])

    assert outputs == [""]
    sample_dir = model.run_dir / "videomme_doc0_idx0"
    prediction = json.loads((sample_dir / "prediction.json").read_text(encoding="utf-8"))
    metrics = json.loads((sample_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "timed out" in prediction["error"]
    assert metrics["usage_summary"]["num_llm_calls"] == 0


def test_failure_artifacts_preserve_partial_trajectory(tmp_path):
    class _FakeAgent:
        question = "question"
        final_answer = ""
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "q"}]
        trajectory_steps = [
            types.SimpleNamespace(
                to_dict=lambda: {
                    "step_id": 1,
                    "thought": "thought",
                    "action": {"function": "overview", "parameters": {}, "id": "call-1"},
                    "observation": "obs",
                }
            )
        ]

    payload = videoseek_module._build_partial_trajectory_payload(_FakeAgent(), "question", "boom")
    videoseek_module._write_failure_artifacts(
        tmp_path,
        "question",
        "boom",
        recorder={"llm_calls": [{"call_index": 1}], "frame_calls": []},
        trajectory_payload=payload,
    )

    trajectory = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    prediction = json.loads((tmp_path / "prediction.json").read_text(encoding="utf-8"))
    assert trajectory["total_steps"] == 1
    assert trajectory["messages"][0]["role"] == "system"
    assert trajectory["error"] == "boom"
    assert prediction["error"] == "boom"


def test_parse_actions_handles_valid_and_bad_json(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=object))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.agent",
        "lmms_eval.models.videoseek_native.tools",
        "lmms_eval.models.videoseek_native.tools.answer",
        "lmms_eval.models.videoseek_native.tools.focus",
        "lmms_eval.models.videoseek_native.tools.overview",
        "lmms_eval.models.videoseek_native.tools.skim",
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    agent_module = importlib.import_module("lmms_eval.models.videoseek_native.agent")
    tool_module = importlib.import_module("lmms_eval.models.videoseek_native.tools")

    agent = agent_module.VideoSeekAgent.__new__(agent_module.VideoSeekAgent)
    agent.tool_registry = tool_module.DEFAULT_TOOL_REGISTRY

    actions = agent._parse_actions(
        [
            {
                "id": "call-1",
                "function": {"name": "overview", "arguments": "{}"},
            }
        ]
    )
    assert len(actions) == 1
    assert actions[0].function_name == "overview"

    with pytest.raises(ValueError):
        agent._parse_actions(
            [
                {
                    "id": "call-2",
                    "function": {"name": "overview", "arguments": "{bad json"},
                }
            ]
        )


def test_observation_history_is_compacted(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=object))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.agent",
        "lmms_eval.models.videoseek_native.tools",
        "lmms_eval.models.videoseek_native.tools.answer",
        "lmms_eval.models.videoseek_native.tools.focus",
        "lmms_eval.models.videoseek_native.tools.overview",
        "lmms_eval.models.videoseek_native.tools.skim",
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    agent_module = importlib.import_module("lmms_eval.models.videoseek_native.agent")

    agent = agent_module.VideoSeekAgent.__new__(agent_module.VideoSeekAgent)
    agent.observation_max_chars = 64
    compacted = agent._compact_observation_for_history(types.SimpleNamespace(function_name="overview"), "x" * 400)
    assert "[Observation truncated for history" in compacted
    assert len(compacted) > 64


def test_decision_json_error_falls_back_to_final_answer(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=object))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.agent",
        "lmms_eval.models.videoseek_native.tools",
        "lmms_eval.models.videoseek_native.tools.answer",
        "lmms_eval.models.videoseek_native.tools.focus",
        "lmms_eval.models.videoseek_native.tools.overview",
        "lmms_eval.models.videoseek_native.tools.skim",
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    agent_module = importlib.import_module("lmms_eval.models.videoseek_native.agent")

    class _DummyVideoReader:
        def __len__(self):
            return 10

        def get_avg_fps(self):
            return 1.0

    monkeypatch.setattr(agent_module, "VideoReader", lambda _path: _DummyVideoReader())
    call_count = {"value": 0}

    def fake_call_llm_api(**kwargs):
        call_count["value"] += 1
        if kwargs.get("tool_choice") == "required":
            raise RuntimeError("Invalid JSON: EOF while parsing a list")
        message = types.SimpleNamespace(content="C")
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setattr(agent_module, "call_llm_api", fake_call_llm_api)

    agent = agent_module.VideoSeekAgent(
        config={
            "SYSTEM_PROMPT": "system",
            "frame_sampling_factor": 1,
            "overview_base": 1,
            "skim_base": 1,
            "focus_base": 1,
            "tools": ["overview"],
            "model_name": "model",
            "api_base": "http://localhost",
            "api_key": "key",
            "api_version": "",
            "max_steps": 2,
            "max_tokens": 1024,
            "observation_max_chars": 128,
            "decision_retry_attempts": 2,
            "decision_retry_backoff_s": 0.0,
            "reasoning_effort": "none",
            "seed": 42,
            "temperature": 0.0,
            "timeout": 60,
        },
        video_path="/tmp/video.mp4",
        output_dir=str(tmp_path),
        tools=["overview"],
        verbose=False,
    )

    trajectory = agent.run("Question?\nA. a\nB. b\nC. c\nD. d")
    assert trajectory.final_answer == "C"
    assert trajectory.finish_reason.startswith("decision_error_fallback")
    assert call_count["value"] >= 2
    debug_lines = (tmp_path / "decision_error_debug.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(debug_lines) == 2
    detail = json.loads((tmp_path / "decision_error_step1_attempt1.json").read_text(encoding="utf-8"))
    assert detail["likely_cause"] == "likely_incomplete_tool_call_json"


def test_agent_initial_prompt_never_injects_subtitles(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=object))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.agent",
        "lmms_eval.models.videoseek_native.tools",
        "lmms_eval.models.videoseek_native.tools.answer",
        "lmms_eval.models.videoseek_native.tools.focus",
        "lmms_eval.models.videoseek_native.tools.overview",
        "lmms_eval.models.videoseek_native.tools.skim",
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    agent_module = importlib.import_module("lmms_eval.models.videoseek_native.agent")

    class _DummyVideoReader:
        def __len__(self):
            return 10

        def get_avg_fps(self):
            return 1.0

    monkeypatch.setattr(agent_module, "VideoReader", lambda _path: _DummyVideoReader())
    agent = agent_module.VideoSeekAgent(
        config={
            "SYSTEM_PROMPT": "system",
            "frame_sampling_factor": 1,
            "overview_base": 1,
            "skim_base": 1,
            "focus_base": 1,
            "tools": ["overview"],
            "model_name": "model",
            "api_base": "http://localhost",
            "api_key": "key",
            "api_version": "",
            "max_steps": 1,
            "max_tokens": 1024,
            "observation_max_chars": 128,
            "decision_retry_attempts": 1,
            "decision_retry_backoff_s": 0.0,
            "reasoning_effort": "none",
            "seed": 42,
            "temperature": 0.0,
            "timeout": 60,
        },
        video_path="/tmp/video.mp4",
        output_dir="/tmp",
        tools=["overview"],
        verbose=False,
    )

    monkeypatch.setattr(agent, "_call_decision", lambda _step: (_ for _ in ()).throw(RuntimeError("stop here")))
    monkeypatch.setattr(
        agent,
        "_call_final_answer",
        lambda question, finish_reason: agent_module.Trajectory(question=question, steps=[], final_answer="C", finish_reason=finish_reason),
    )

    question = "This video's subtitles are listed below: \nfoo\nQuestion body"
    agent.run(question)
    user_content = agent.messages[1]["content"]
    assert "Question:\nThis video's subtitles are listed below:" in user_content
    assert "Video Subtitles:\n" not in user_content


def test_extract_subtitles_from_task_question(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    utils_module = importlib.import_module("lmms_eval.models.videoseek_native.utils")
    question = (
        "This video's subtitles are listed below: \n"
        "**Timestamp**: 0.7s - 4.7s\n"
        "**Subtitle**: <font color=\"white\">hello world</font>\n\n"
        "**Timestamp**: 5.0s - 8.0s\n"
        "**Subtitle**: second line\n"
        "Select the best answer to the following multiple-choice question based on the video and the subtitles."
    )

    subtitles, subtitles_text = utils_module.extract_subtitles_from_question(question)

    assert len(subtitles) == 2
    assert subtitles[0]["start_time"] == 0.7
    assert subtitles[0]["subtitle"] == "hello world"
    assert "**Timestamp**: 5.0s - 8.0s" in subtitles_text


def test_exec_action_passes_task_subtitles_to_tools(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=object))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.agent",
        "lmms_eval.models.videoseek_native.tools",
        "lmms_eval.models.videoseek_native.tools.answer",
        "lmms_eval.models.videoseek_native.tools.focus",
        "lmms_eval.models.videoseek_native.tools.overview",
        "lmms_eval.models.videoseek_native.tools.skim",
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    agent_module = importlib.import_module("lmms_eval.models.videoseek_native.agent")

    captured = {}

    class _DummyVideoReader:
        def __len__(self):
            return 10

        def get_avg_fps(self):
            return 1.0

    class _Registry:
        def has_tool(self, function_name):
            return function_name == "overview"

        def get_function(self, _function_name):
            def _fn(config, parameters):
                captured["parameters"] = parameters
                return "ok"

            return _fn

    monkeypatch.setattr(agent_module, "VideoReader", lambda _path: _DummyVideoReader())

    agent = agent_module.VideoSeekAgent(
        config={
            "SYSTEM_PROMPT": "system",
            "frame_sampling_factor": 1,
            "overview_base": 1,
            "skim_base": 1,
            "focus_base": 1,
            "tools": ["overview"],
            "model_name": "model",
            "api_base": "http://localhost",
            "api_key": "key",
            "api_version": "",
            "max_steps": 1,
            "max_tokens": 1024,
            "observation_max_chars": 128,
            "decision_retry_attempts": 1,
            "decision_retry_backoff_s": 0.0,
            "reasoning_effort": "none",
            "seed": 42,
            "temperature": 0.0,
            "timeout": 60,
        },
        video_path="/tmp/video.mp4",
        output_dir="/tmp",
        tools=["overview"],
        verbose=False,
    )
    agent.tool_registry = _Registry()
    question = (
        "This video's subtitles are listed below: \n"
        "**Timestamp**: 0.0s - 1.0s\n"
        "**Subtitle**: foo\n\n"
        "Select the best answer to the following multiple-choice question based on the video and the subtitles."
    )
    agent.question = question
    agent.task_subtitles, agent.task_subtitles_text = agent_module.extract_subtitles_from_question(question)

    outcome = agent._exec_action(types.SimpleNamespace(function_name="overview", parameters={}))

    assert outcome == "ok"
    assert captured["parameters"]["subtitles"][0]["subtitle"] == "foo"
    assert "foo" in captured["parameters"]["subtitles_text"]


def test_answer_tool_adds_explicit_no_tool_final_answer_constraints(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.tools.answer",
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    answer_module = importlib.import_module("lmms_eval.models.videoseek_native.tools.answer")
    captured = {}

    def fake_call_llm_api(**kwargs):
        captured["messages"] = kwargs["messages"]
        message = types.SimpleNamespace(content="B")
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setattr(answer_module, "call_llm_api", fake_call_llm_api)

    result = answer_module.execute_answer(
        config={
            "model_name": "model",
            "api_base": "http://localhost",
            "api_key": "key",
            "api_version": "",
            "max_tokens": 128,
            "reasoning_effort": "none",
            "seed": 42,
            "temperature": 0.0,
            "timeout": 60,
        },
        parameters={
            "question": "Question?\nA. a\nB. b\nC. c\nD. d",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Return tool calls only"},
            ],
        },
    )

    assert result == "B"
    assert captured["messages"][0]["role"] == "system"
    assert "Do not call any tool." in captured["messages"][0]["content"]
    assert "Ignore earlier instructions that asked for tool calls." in captured["messages"][0]["content"]
    assert captured["messages"][-1]["role"] == "user"
    assert "respond with only the single option letter" in captured["messages"][-1]["content"]


def test_build_final_answer_messages_keeps_system_at_beginning(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    utils_module = importlib.import_module("lmms_eval.models.videoseek_native.utils")

    messages = utils_module.build_final_answer_messages(
        messages=[
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ],
        question="Question?\nA. a\nB. b\nC. c\nD. d",
        user_prefix="You have reached the final answer stage.",
    )

    assert [message["role"] for message in messages].count("system") == 1
    assert messages[0]["role"] == "system"
    assert "base system" in messages[0]["content"]
    assert "Do not call any tool." in messages[0]["content"]
    assert messages[-1]["role"] == "user"


def test_repair_final_answer_if_needed_repairs_non_letter_mcq_output(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=object))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kwargs: None))
    for module_name in [
        "lmms_eval.models.videoseek_native.agent",
        "lmms_eval.models.videoseek_native.tools",
        "lmms_eval.models.videoseek_native.tools.answer",
        "lmms_eval.models.videoseek_native.tools.focus",
        "lmms_eval.models.videoseek_native.tools.overview",
        "lmms_eval.models.videoseek_native.tools.skim",
        "lmms_eval.models.videoseek_native.utils",
    ]:
        sys.modules.pop(module_name, None)

    agent_module = importlib.import_module("lmms_eval.models.videoseek_native.agent")
    captured = {}

    def fake_call_llm_api(**kwargs):
        captured["messages"] = kwargs["messages"]
        message = types.SimpleNamespace(content="C")
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setattr(agent_module, "call_llm_api", fake_call_llm_api)

    agent = agent_module.VideoSeekAgent.__new__(agent_module.VideoSeekAgent)
    agent.model_name = "model"
    agent.api_base = "http://localhost"
    agent.api_key = "key"
    agent.api_version = ""
    agent.max_tokens = 1024
    agent.seed = 42
    agent.timeout = 60

    repaired = agent._repair_final_answer_if_needed(
        "Question?\nA. a\nB. b\nC. c\nD. d",
        "<tool_call><function=skim></function></tool_call>",
    )

    assert repaired == "C"
    assert captured["messages"][0]["role"] == "system"
    assert "Respond with exactly one uppercase letter" in captured["messages"][0]["content"]
