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
