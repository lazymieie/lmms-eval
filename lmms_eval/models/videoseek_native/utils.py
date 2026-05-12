import os
import random
import re
import time
from contextlib import contextmanager
from threading import local

from litellm import completion


_THREAD_STATE = local()


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                error_text = str(exc)
                if (
                    "rate limit" in error_text.lower()
                    or "timed out" in error_text
                    or "Too Many Requests" in error_text
                    or "Forbidden for url" in error_text
                    or "the maximum usage" in error_text.lower()
                    or "server had an error" in error_text.lower()
                    or "internal" in error_text.lower()
                ):
                    num_retries += 1
                    if num_retries > max_retries:
                        raise RuntimeError(f"Max retries reached for LiteLLM request: {error_text}") from exc
                    delay *= exponential_base * (1 + jitter * random.random())
                    time.sleep(delay)
                else:
                    raise

    return wrapper


def _get_thread_recorder() -> dict | None:
    return getattr(_THREAD_STATE, "recorder", None)


def set_thread_recorder(recorder: dict | None) -> None:
    _THREAD_STATE.recorder = recorder


def get_call_label() -> str | None:
    return getattr(_THREAD_STATE, "call_label", None)


def set_call_label(label: str | None) -> None:
    _THREAD_STATE.call_label = label


@contextmanager
def call_label(label: str):
    previous = get_call_label()
    set_call_label(label)
    try:
        yield
    finally:
        set_call_label(previous)


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


def _infer_call_type(messages, tool_choice) -> str:
    explicit = get_call_label()
    if explicit:
        return explicit
    if tool_choice == "required":
        return "decision"
    if messages:
        last_text = _message_text(messages[-1])
        if "You have reached the maximum number of steps." in last_text:
            return "final_answer_fallback"
    return "llm_call"


def _record_llm_call(messages, response, error, started_at, tool_choice) -> None:
    recorder = _get_thread_recorder()
    if recorder is None:
        return

    usage = getattr(response, "usage", None) if response is not None else None
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    reasoning_tokens = 0
    if usage is not None:
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        else:
            reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0)

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
            "call_type": _infer_call_type(messages or [], tool_choice),
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


def record_frame_event(event: dict) -> None:
    recorder = _get_thread_recorder()
    if recorder is not None:
        recorder["frame_calls"].append(event)


@retry_with_exponential_backoff
def call_llm_api(
    model_name: str,
    messages: list,
    api_base: str,
    api_key: str = None,
    api_version: str = None,
    max_tokens: int = 4096,
    reasoning_effort: str = "medium",
    seed: int = 42,
    temperature: float = 1.0,
    tools: list = None,
    tool_choice: str = None,
    return_json: bool = False,
    timeout: int = 900,
):
    started_at = time.time()
    response = None
    error = None
    try:
        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "api_base": api_base,
            "api_key": api_key,
            "api_version": api_version,
            "max_completion_tokens": max_tokens,
            "seed": seed,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": {"type": "json_object"} if return_json else None,
            "timeout": timeout,
        }
        if reasoning_effort not in (None, "", "none"):
            request_kwargs["reasoning_effort"] = reasoning_effort
        try:
            response = completion(**request_kwargs)
        except Exception as exc:
            error_text = str(exc)
            if "reasoning_effort" in error_text and "does not support parameters" in error_text and "reasoning_effort" in request_kwargs:
                request_kwargs.pop("reasoning_effort", None)
                response = completion(**request_kwargs)
            else:
                raise
        return response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        _record_llm_call(messages, response, error, started_at, tool_choice)


def load_subtitles(subtitle_path: str):
    """Parse SRT file and return list of {start_time, end_time, subtitle} dicts."""
    if subtitle_path is None or not os.path.exists(subtitle_path):
        return []
    with open(subtitle_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    result = []
    pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
    blocks = re.split(r"\n\n+", content.strip())

    def to_seconds(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for block in blocks:
        match = pattern.search(block)
        if match:
            start = to_seconds(*match.groups()[:4])
            end = to_seconds(*match.groups()[4:8])
            text = block[match.end() :].strip().replace("\n", " ")
            result.append({"start_time": round(start, 1), "end_time": round(end, 1), "subtitle": text})
    return result


def convert_to_free_form_text_representation(history: list[dict], content_type: str = "caption") -> str:
    free_form_text_representation = ""
    if len(history) == 0:
        return f"No {content_type} found."
    for item in history:
        if item[content_type] is None:
            continue
        start_time, end_time = item["start_time"], item["end_time"]
        free_form_text_representation += (
            f"**Timestamp**: {start_time}s - {end_time}s\n"
            f"**{content_type.capitalize()}**: {item[content_type]}\n\n"
        )
    return free_form_text_representation


def extract_mcq_letter(text: str) -> str:
    match = re.search(r"\b([A-D])\b", str(text).upper())
    return match.group(1) if match else str(text)
