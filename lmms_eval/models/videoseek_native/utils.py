import os
import random
import re
import time
from contextlib import contextmanager
from itertools import count
from threading import Lock
from threading import local

from litellm import completion


_THREAD_STATE = local()
_API_BASE_ROUND_ROBIN = count()
_API_BASE_LOCK = Lock()


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        retry_max_retries = max(0, int(kwargs.pop("_retry_max_retries", max_retries)))
        retry_initial_delay = max(0.0, float(kwargs.pop("_retry_initial_delay", initial_delay)))
        retry_exponential_base = max(1.0, float(kwargs.pop("_retry_exponential_base", exponential_base)))
        retry_jitter = kwargs.pop("_retry_jitter", jitter)

        num_retries = 0
        delay = retry_initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                error_text = str(exc)
                error_text_lower = error_text.lower()
                if (
                    "rate limit" in error_text_lower
                    or "timed out" in error_text
                    or "timeout" in error_text_lower
                    or "Too Many Requests" in error_text
                    or "Forbidden for url" in error_text
                    or "the maximum usage" in error_text_lower
                    or "server had an error" in error_text_lower
                    or "internal" in error_text_lower
                ):
                    num_retries += 1
                    if num_retries > retry_max_retries:
                        raise RuntimeError(f"Max retries reached for LiteLLM request: {error_text}") from exc
                    delay *= retry_exponential_base * (1 + retry_jitter * random.random())
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


def normalize_api_bases(api_base) -> list[str]:
    if api_base is None:
        return []
    if isinstance(api_base, (list, tuple)):
        values = [str(value).strip() for value in api_base if str(value).strip()]
        return values
    raw = str(api_base).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            import json

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(value).strip() for value in parsed if str(value).strip()]
        except Exception:
            pass
    delimiter = "|" if "|" in raw else ","
    return [value.strip() for value in raw.split(delimiter) if value.strip()]


def select_api_base(api_base) -> str:
    api_bases = normalize_api_bases(api_base)
    if not api_bases:
        return str(api_base or "").strip()
    if len(api_bases) == 1:
        return api_bases[0]
    with _API_BASE_LOCK:
        index = next(_API_BASE_ROUND_ROBIN) % len(api_bases)
    return api_bases[index]


FINAL_ANSWER_SYSTEM_TEXT = (
    "You are now in the final answer stage. "
    "Do not call any tool. "
    "Do not output XML, JSON, code fences, or any tool-call syntax. "
    "Ignore earlier instructions that asked for tool calls. "
    "Answer directly using the requested final answer format only."
)


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
    api_base,
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
        selected_api_base = select_api_base(api_base)
        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "api_base": selected_api_base,
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


def build_final_answer_messages(messages: list, question: str, user_prefix: str) -> list:
    prepared_messages = [dict(message) for message in (messages or [])]
    if prepared_messages and prepared_messages[0].get("role") == "system":
        merged_system = dict(prepared_messages[0])
        merged_system["content"] = (
            f"{str(merged_system.get('content', '')).rstrip()}\n\n{FINAL_ANSWER_SYSTEM_TEXT}"
        ).strip()
        prepared_messages[0] = merged_system
    else:
        prepared_messages.insert(0, {"role": "system", "content": FINAL_ANSWER_SYSTEM_TEXT})

    prepared_messages.append(
        {
            "role": "user",
            "content": (
                f"{user_prefix}\n"
                f"Question:\n{question}\n\n"
                "Provide the final answer now. "
                "Do not call any tool and do not propose further actions. "
                "If this is a multiple-choice question, respond with only the single option letter from the given choices."
            ),
        }
    )
    return prepared_messages


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


def extract_subtitles_from_question(question: str) -> tuple[list[dict], str]:
    header = "This video's subtitles are listed below:"
    if not question or header not in question:
        return [], ""

    start_idx = question.index(header) + len(header)
    subtitle_block = question[start_idx:]
    end_markers = [
        "\nSelect the best answer to the following multiple-choice question based on the video",
        "\nQuestion:",
        "\nOptions:",
    ]
    end_positions = [subtitle_block.find(marker) for marker in end_markers if subtitle_block.find(marker) != -1]
    if end_positions:
        subtitle_block = subtitle_block[: min(end_positions)]
    subtitle_block = subtitle_block.strip()

    if not subtitle_block or subtitle_block == "No subtitles available":
        return [], ""

    pattern = re.compile(
        r"\*\*Timestamp\*\*:\s*([0-9.]+)s\s*-\s*([0-9.]+)s\s*\n\*\*Subtitle\*\*:\s*(.+?)(?=\n\*\*Timestamp\*\*:\s*|\Z)",
        re.DOTALL,
    )
    subtitles = []
    for match in pattern.finditer(subtitle_block):
        text = re.sub(r"<[^>]+>", "", match.group(3)).strip()
        subtitles.append(
            {
                "start_time": float(match.group(1)),
                "end_time": float(match.group(2)),
                "subtitle": text,
            }
        )

    if subtitles:
        return subtitles, convert_to_free_form_text_representation(subtitles, content_type="subtitle").strip()

    raw_text = re.sub(r"<[^>]+>", "", subtitle_block).strip()
    return [], raw_text


def subtitles_prompt_text(
    subtitles: list[dict],
    subtitles_text: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str:
    if subtitles:
        filtered = subtitles
        if start_time is not None and end_time is not None:
            filtered = [
                subtitle
                for subtitle in subtitles
                if float(subtitle["start_time"]) <= float(end_time) and float(subtitle["end_time"]) >= float(start_time)
            ]
        if filtered:
            return convert_to_free_form_text_representation(filtered, content_type="subtitle").strip()
        return ""
    return (subtitles_text or "").strip()


def extract_mcq_letter(text: str) -> str:
    match = re.search(r"\b([A-D])\b", str(text).upper())
    return match.group(1) if match else str(text)
