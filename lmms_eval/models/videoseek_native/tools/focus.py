import base64
from io import BytesIO

import numpy as np
from PIL import Image

from ..utils import call_label, call_llm_api, record_frame_event, subtitles_prompt_text


focus_tool = {
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
}


def execute_focus(config: dict, parameters: dict) -> str:
    query = parameters["query"]
    start_time = parameters["start_time"]
    end_time = parameters["end_time"]
    max_num_frames = config["frame_sampling_factor"] * config["focus_base"]
    vr = parameters["vr"]
    subtitles = parameters.get("subtitles", [])
    subtitles_text = subtitles_prompt_text(subtitles, parameters.get("subtitles_text", ""), start_time=start_time, end_time=end_time)

    start_frame = int(start_time * vr.get_avg_fps())
    end_frame = min(int(end_time * vr.get_avg_fps()), len(vr) - 1)
    num_frames = max(1, min(int(end_frame - start_frame), max_num_frames))
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).astype(int)
    cur_timestamps = np.array([round(frame_index / vr.get_avg_fps(), 1) for frame_index in frame_indices], dtype=np.float32)
    frames = vr.get_batch(frame_indices).asnumpy()

    content = [{"type": "text", "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s):\n"}]
    for frame, timestamp in zip(frames, cur_timestamps):
        image = Image.fromarray(frame)
        output_buffer = BytesIO()
        image.save(output_buffer, format="jpeg")
        base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        content.append({"type": "text", "text": f"{timestamp:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    content.append(
        {
            "type": "text",
            "text": (
                (f"Video Subtitles:\n{subtitles_text}\n\n" if subtitles_text else "")
                +
                f"Question:\n{query}\n\n"
                "Please answer the question based on the given video clip in at most 80 words. "
                "If the clip is not related to the question, please return 'No relevant content found.'"
            ),
        }
    )

    record_frame_event(
        {
            "tool_name": "focus",
            "frames_sampled": int(num_frames),
            "image_inputs": int(num_frames),
            "start_time": start_time,
            "end_time": end_time,
        }
    )
    with call_label("focus"):
        response = call_llm_api(
            messages=[{"role": "user", "content": content}],
            model_name=config["model_name"],
            api_base=config["api_base"],
            api_key=config["api_key"],
            api_version=config["api_version"],
            max_tokens=config["max_tokens"],
            reasoning_effort=config.get("tool_reasoning_effort", "none"),
            seed=config["seed"],
            temperature=config["temperature"],
            timeout=config.get("timeout", 900),
        )

    return response.choices[0].message.content
