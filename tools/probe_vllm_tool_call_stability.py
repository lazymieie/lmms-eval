#!/usr/bin/env python3
"""
Minimal probe for OpenAI-compatible tool calling stability.

Purpose:
- Isolate vLLM tool-call parsing from VideoSeek.
- Send a trivial single-tool request repeatedly with `tool_choice="required"`.
- Save the raw HTTP response body so parser-induced corruption is visible.

Typical usage:
    python tools/probe_vllm_tool_call_stability.py \
      --api-base http://10.233.122.95:5590/v1 \
      --model Qwen3.5-27B \
      --trials 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenAI-compatible tool-call stability.")
    parser.add_argument("--api-base", required=True, help="Base URL, e.g. http://host:5590/v1")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--api-key", default="any", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--trials", type=int, default=20, help="Number of repeated requests.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=256, help="max_tokens for the request.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write raw responses and summary. Defaults to logs/tool_call_probe/<timestamp>.",
    )
    parser.add_argument(
        "--prompt",
        default="Call the ping tool with value='ok'. Return exactly one tool call.",
        help="User prompt for the probe.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "tool_call_probe" / f"run_{timestamp}"


def build_payload(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a tool-calling assistant. "
                    "When tools are available and tool_choice is required, return exactly one valid tool call."
                ),
            },
            {"role": "user", "content": args.prompt},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Return a trivial ping marker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "Must be exactly 'ok'.",
                            }
                        },
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }


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


def classify_response(raw_body: str) -> dict:
    result = {
        "ok": False,
        "reason": "",
        "tool_name": None,
        "arguments_text": None,
        "arguments_json_ok": False,
    }
    try:
        payload = json.loads(raw_body)
    except Exception as exc:
        result["reason"] = f"response_body_not_json: {type(exc).__name__}: {exc}"
        return result

    choices = payload.get("choices") or []
    if not choices:
        result["reason"] = "missing_choices"
        return result

    message = (choices[0] or {}).get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) != 1:
        result["reason"] = f"expected_one_tool_call_got_{len(tool_calls)}"
        return result

    function = (tool_calls[0] or {}).get("function") or {}
    result["tool_name"] = function.get("name")
    result["arguments_text"] = function.get("arguments")
    if result["tool_name"] != "ping":
        result["reason"] = f"unexpected_tool_name:{result['tool_name']}"
        return result

    try:
        arguments = json.loads(result["arguments_text"] or "")
    except Exception as exc:
        result["reason"] = f"arguments_not_json: {type(exc).__name__}: {exc}"
        return result

    result["arguments_json_ok"] = True
    if arguments.get("value") != "ok":
        result["reason"] = f"unexpected_argument_value:{arguments.get('value')!r}"
        return result

    result["ok"] = True
    result["reason"] = "ok"
    return result


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = args.api_base.rstrip("/") + "/chat/completions"
    payload = build_payload(args)
    (output_dir / "request_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_path = output_dir / "raw_responses.jsonl"
    summary_path = output_dir / "summary.json"

    successes = 0
    failures = 0
    counts_by_reason: dict[str, int] = {}
    started_at = time.time()

    with raw_path.open("w", encoding="utf-8") as handle:
        for trial in range(1, args.trials + 1):
            trial_started = time.time()
            status_code, raw_body = post_json(endpoint, args.api_key, payload, args.timeout)
            elapsed = time.time() - trial_started
            classification = classify_response(raw_body)
            record = {
                "trial": trial,
                "status_code": status_code,
                "elapsed_s": round(elapsed, 3),
                "classification": classification,
                "raw_body": raw_body,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if classification["ok"]:
                successes += 1
                print(f"[{trial}/{args.trials}] ok elapsed={elapsed:.2f}s")
            else:
                failures += 1
                reason = classification["reason"]
                counts_by_reason[reason] = counts_by_reason.get(reason, 0) + 1
                print(f"[{trial}/{args.trials}] fail elapsed={elapsed:.2f}s reason={reason}")

    summary = {
        "api_base": args.api_base,
        "model": args.model,
        "trials": args.trials,
        "successes": successes,
        "failures": failures,
        "success_rate": (successes / args.trials) if args.trials else 0.0,
        "counts_by_reason": counts_by_reason,
        "elapsed_s_total": round(time.time() - started_at, 3),
        "request_payload_file": str(output_dir / "request_payload.json"),
        "raw_responses_file": str(raw_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
