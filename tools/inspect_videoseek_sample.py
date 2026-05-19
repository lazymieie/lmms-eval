#!/usr/bin/env python3
"""Inspect one native VideoSeek sample directory and summarize likely failure causes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def truncate(text: str, limit: int = 240) -> str:
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize_llm_calls(metrics: dict[str, Any] | None) -> dict[str, Any]:
    calls = (metrics or {}).get("llm_calls", []) or []
    if not calls:
        return {"num_llm_calls": 0}
    last = calls[-1]
    longest = max(calls, key=lambda item: float(item.get("elapsed_s", 0) or 0))
    return {
        "num_llm_calls": len(calls),
        "last_call_type": last.get("call_type"),
        "last_call_elapsed_s": last.get("elapsed_s"),
        "last_call_error": last.get("error"),
        "last_call_finish_reason": last.get("finish_reason"),
        "last_call_message_count": last.get("message_count"),
        "longest_call_type": longest.get("call_type"),
        "longest_call_elapsed_s": longest.get("elapsed_s"),
        "longest_call_error": longest.get("error"),
    }


def load_last_decision_debug(sample_dir: Path) -> dict[str, Any] | None:
    files = sorted(sample_dir.glob("decision_error_step*_attempt*.json"))
    if not files:
        return None

    def sort_key(path: Path) -> tuple[int, int]:
        match = re.search(r"step(\d+)_attempt(\d+)", path.name)
        if not match:
            return (0, 0)
        return (int(match.group(1)), int(match.group(2)))

    latest = sorted(files, key=sort_key)[-1]
    payload = load_json(latest) or {}
    payload["_file"] = str(latest)
    return payload


def infer_root_cause(
    prediction: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    trajectory: dict[str, Any] | None,
    decision_debug: dict[str, Any] | None,
) -> list[str]:
    hints: list[str] = []
    error_text = str((prediction or {}).get("error", "") or (metrics or {}).get("error", "") or "").lower()
    llm_calls = (metrics or {}).get("llm_calls", []) or []
    frame_calls = (metrics or {}).get("tool_frame_calls", []) or []

    if "timed out after" in error_text:
        hints.append("sample hit the outer per-sample timeout in the parent process")
        if decision_debug is not None:
            hints.append(
                "child process had already reached a decision JSON/tool-call failure before timeout"
            )
            likely = decision_debug.get("likely_cause")
            if likely:
                hints.append(f"latest decision failure hint: {likely}")
        elif llm_calls:
            last = llm_calls[-1]
            call_type = last.get("call_type")
            elapsed = last.get("elapsed_s")
            hints.append(f"last recorded LLM call before timeout: {call_type} ({elapsed}s)")
            if call_type == "decision":
                hints.append("timeout likely happened while generating/parsing the next tool call")
            elif call_type in {"overview", "skim", "focus", "answer"}:
                hints.append("timeout likely happened inside a tool model call rather than in tool-call parsing")
        elif frame_calls:
            hints.append("some frame tools had run, but no LLM call metrics were recorded")
        else:
            hints.append("no inner agent progress was recorded before the parent timeout")

    if decision_debug is not None:
        line = ((decision_debug.get("error_location") or {}).get("line")) or 0
        likely = str(decision_debug.get("likely_cause", ""))
        if "overlong" in likely or int(line or 0) > 10000:
            hints.append("decision output likely ran away into an overlong malformed tool-call response")

    if trajectory is not None and not (trajectory.get("steps") or []):
        hints.append("no completed trajectory steps were persisted")

    return hints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=Path, help="Path to a native VideoSeek sample directory")
    args = parser.parse_args()

    sample_dir = args.sample_dir.expanduser().resolve()
    if not sample_dir.is_dir():
        raise SystemExit(f"Sample directory not found: {sample_dir}")

    prediction = load_json(sample_dir / "prediction.json")
    metrics = load_json(sample_dir / "metrics.json")
    trajectory = load_json(sample_dir / "trajectory.json")
    decision_debug = load_last_decision_debug(sample_dir)

    summary = {
        "sample_dir": str(sample_dir),
        "prediction": {
            "prediction": (prediction or {}).get("prediction", ""),
            "error": (prediction or {}).get("error", ""),
        },
        "trajectory": {
            "total_steps": (trajectory or {}).get("total_steps"),
            "finish_reason": (trajectory or {}).get("finish_reason"),
            "error": (trajectory or {}).get("error"),
            "message_count": (trajectory or {}).get("message_count"),
        },
        "llm_calls": summarize_llm_calls(metrics),
        "usage_summary": (metrics or {}).get("usage_summary", {}),
        "frame_summary": (metrics or {}).get("frame_summary", {}),
        "latest_decision_debug": None,
        "root_cause_hints": [],
    }

    if decision_debug is not None:
        summary["latest_decision_debug"] = {
            "file": decision_debug.get("_file"),
            "step": decision_debug.get("step"),
            "attempt": decision_debug.get("attempt"),
            "error_type": decision_debug.get("error_type"),
            "likely_cause": decision_debug.get("likely_cause"),
            "error_location": decision_debug.get("error_location"),
            "message_count": decision_debug.get("message_count"),
            "trajectory_steps_count": decision_debug.get("trajectory_steps_count"),
            "max_tokens": decision_debug.get("max_tokens"),
            "total_message_text_chars": decision_debug.get("total_message_text_chars"),
            "last_message_preview": truncate(decision_debug.get("last_message_preview", "")),
            "error": truncate(decision_debug.get("error", ""), 400),
        }

    summary["root_cause_hints"] = infer_root_cause(prediction, metrics, trajectory, decision_debug)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
