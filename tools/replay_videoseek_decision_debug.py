#!/usr/bin/env python3
"""
Replay a saved VideoSeek decision-error snapshot against an OpenAI-compatible API.

Purpose:
- Reproduce failures from `decision_error_step*_attempt*.json`
- Capture the raw HTTP response body from vLLM
- Distinguish:
  1. response body is not valid JSON
  2. response body JSON is valid but `message.tool_calls` is malformed/missing
  3. model falls back to text tool-call syntax like `<tool_call>...</tool_call>`

Typical usage:
    python tools/replay_videoseek_decision_debug.py \
      --debug-file /path/to/decision_error_step4_attempt1.json \
      --api-base http://10.233.122.95:5590/v1 \
      --model Qwen3.5-27B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "overview",
            "description": "To get a structured video summary for the entire video.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skim",
            "description": "To localize moments related to the query by scanning a longer time range with sparse frame sampling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A concise question to verify within the video segment."},
                    "start_time": {"type": "number", "description": "The start time of the video to skim."},
                    "end_time": {"type": "number", "description": "The end time of the video to skim."},
                },
                "required": ["query", "start_time", "end_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus",
            "description": "To verify fine visual details with dense inspection of a short clip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A concise question to verify within the short clip."},
                    "start_time": {"type": "number", "description": "The start time of the video to focus."},
                    "end_time": {"type": "number", "description": "The end time of the video to focus."},
                },
                "required": ["query", "start_time", "end_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Based on the given trajectory, generate the final answer to the question.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a VideoSeek decision-error snapshot.")
    parser.add_argument("--debug-file", required=True, help="Path to decision_error_step*.json")
    parser.add_argument("--api-base", required=True, help="Base URL, e.g. http://host:5590/v1")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--api-key", default="any", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens. Defaults to the value recorded in the debug file.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional reasoning_effort override. If omitted, do not send this field.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write replay artifacts. Defaults beside the debug file.",
    )
    return parser.parse_args()


def post_json(url: str, api_key: str, payload: dict, timeout: float) -> tuple[int | None, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.getcode(), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except Exception as exc:  # pragma: no cover - operational probe
        return None, f"REQUEST_EXCEPTION: {type(exc).__name__}: {exc}"


def summarize_response(raw_body: str) -> dict:
    summary = {
        "response_body_json_ok": False,
        "tool_calls_present": False,
        "tool_calls_count": 0,
        "tool_call_arguments_json_ok": [],
        "text_tool_call_present": False,
        "finish_reason": None,
        "content_preview": "",
        "error": "",
    }
    try:
        payload = json.loads(raw_body)
        summary["response_body_json_ok"] = True
    except Exception as exc:
        summary["error"] = f"response_body_not_json: {type(exc).__name__}: {exc}"
        summary["text_tool_call_present"] = "<tool_call>" in raw_body
        summary["content_preview"] = raw_body[:2000]
        return summary

    choices = payload.get("choices") or []
    if not choices:
        summary["error"] = "missing_choices"
        summary["content_preview"] = json.dumps(payload, ensure_ascii=False)[:2000]
        return summary

    choice = choices[0] or {}
    summary["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    summary["content_preview"] = content[:2000]
    summary["text_tool_call_present"] = "<tool_call>" in content

    tool_calls = message.get("tool_calls") or []
    summary["tool_calls_present"] = len(tool_calls) > 0
    summary["tool_calls_count"] = len(tool_calls)
    for tool_call in tool_calls:
        function = (tool_call or {}).get("function") or {}
        arguments_text = function.get("arguments")
        try:
            json.loads(arguments_text or "")
            summary["tool_call_arguments_json_ok"].append(True)
        except Exception:
            summary["tool_call_arguments_json_ok"].append(False)

    if summary["tool_calls_present"]:
        summary["error"] = ""
    elif summary["text_tool_call_present"]:
        summary["error"] = "tool_call_degraded_to_text"
    else:
        summary["error"] = "missing_tool_calls"
    return summary


def main() -> int:
    args = parse_args()
    debug_file = Path(args.debug_file)
    snapshot = json.loads(debug_file.read_text(encoding="utf-8"))
    messages = snapshot["messages"]
    max_tokens = args.max_tokens if args.max_tokens is not None else int(snapshot.get("max_tokens", 4096))

    output_dir = Path(args.output_dir) if args.output_dir else debug_file.parent / f"{debug_file.stem}_replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "seed": args.seed,
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort

    endpoint = args.api_base.rstrip("/") + "/chat/completions"
    request_path = output_dir / "request_payload.json"
    response_path = output_dir / "raw_response.txt"
    summary_path = output_dir / "summary.json"

    request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    started_at = time.time()
    status_code, raw_body = post_json(endpoint, args.api_key, payload, args.timeout)
    elapsed = round(time.time() - started_at, 3)
    response_path.write_text(raw_body, encoding="utf-8")

    summary = {
        "debug_file": str(debug_file),
        "api_base": args.api_base,
        "model": args.model,
        "status_code": status_code,
        "elapsed_s": elapsed,
        "message_count": len(messages),
        "recorded_error": snapshot.get("error"),
        "recorded_error_location": snapshot.get("error_location"),
        "recorded_likely_cause": snapshot.get("likely_cause"),
        "request_payload_file": str(request_path),
        "raw_response_file": str(response_path),
        "response_summary": summarize_response(raw_body),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    response_ok = summary["response_summary"]["response_body_json_ok"]
    has_structured_tool_calls = summary["response_summary"]["tool_calls_present"]
    return 0 if response_ok and has_structured_tool_calls else 1


if __name__ == "__main__":
    sys.exit(main())
