import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image

from ..utils import call_label, call_llm_api, convert_to_free_form_text_representation, record_frame_event


overview_tool = {
    "type": "function",
    "function": {
        "name": "overview",
        "description": "To get a structured video summary for the entire video.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}


def execute_overview(config: dict, parameters: dict) -> str:
    vr = parameters["vr"]
    duration = round(len(vr) / vr.get_avg_fps(), 1)
    subtitles = parameters["subtitles"]
    subtitles_str = convert_to_free_form_text_representation(subtitles, content_type="subtitle")

    num_frames = config["frame_sampling_factor"] * config["overview_base"]
    if num_frames % 8 != 0:
        num_frames = int(np.ceil(num_frames / 8.0) * 8)

    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    cur_timestamps = np.array([round(frame_index / vr.get_avg_fps(), 1) for frame_index in frame_indices], dtype=np.float32)
    cur_timestamps_str = ", ".join([f"{timestamp:.1f}s" for timestamp in cur_timestamps])
    frames = vr.get_batch(frame_indices).asnumpy()

    _, height, width, _ = frames.shape
    short_side = min(height, width)
    scale = 256 / short_side
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    resized_frames = [
        np.array(Image.fromarray(frame).resize((target_width, target_height), Image.BICUBIC))
        for frame in frames
    ]
    frames = np.stack(resized_frames, axis=0)

    group_size = 8
    num_frames, h, w, c = frames.shape
    cur_timestamps = cur_timestamps.reshape(-1, 2, 4)
    num_groups = num_frames // group_size
    frames = frames.reshape(num_groups, 2, 4, h, w, c)
    frames = frames.transpose(0, 1, 3, 2, 4, 5).reshape(num_groups, 2 * h, 4 * w, c)

    content = [{"type": "text", "text": f"The video segment is located at 0.0s - {duration:.1f}s:\nThe video frames are uniformly sampled.\n"}]
    for frame, timestamp in zip(frames, cur_timestamps):
        row_1 = ", ".join([f"{value:.1f}s" for value in timestamp[0]])
        row_2 = ", ".join([f"{value:.1f}s" for value in timestamp[1]])
        timestamp_str = f"[[{row_1}],\n[{row_2}]]"
        image = Image.fromarray(frame)
        output_buffer = BytesIO()
        image.save(output_buffer, format="jpeg")
        base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        content.append({"type": "text", "text": f"[{timestamp[0, 0]:.1f}s - {timestamp[-1, -1]:.1f}s]:\nTimestamp Matrix:\n{timestamp_str}\n"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    content.append(
        {
            "type": "text",
            "text": (
                "Video Subtitles:\n"
                f"{subtitles_str}\n\n"
                "Please generate descriptions for each frame in the video. The descriptions should be concise and detailed (~50 words each).\n"
                f"Ensure every timestamp value exactly matches a timestamp from the provided timestamp matrices (same values and formatting): [{cur_timestamps_str}].\n"
                "Return ONLY valid JSON. Use this exact schema:\n"
                "{\"frames\": [{\"timestamp\": \"1.0s\", \"description\": \"FRAME_DESCRIPTION_1\"}, ...]}\n"
            ),
        }
    )
    record_frame_event({"tool_name": "overview", "frames_sampled": int(num_frames), "image_inputs": int(num_frames // 8)})
    with call_label("overview"):
        response = call_llm_api(
            messages=[{"role": "user", "content": content}],
            model_name=config["model_name"],
            api_base=config["api_base"],
            api_key=config["api_key"],
            api_version=config["api_version"],
            max_tokens=config["max_tokens"],
            reasoning_effort=config["reasoning_effort"],
            seed=config["seed"],
            temperature=config["temperature"],
            return_json=True,
            timeout=config.get("timeout", 900),
        )

    return "\n\n".join([f"{frame['timestamp']}: {frame['description']}" for frame in json.loads(response.choices[0].message.content)["frames"]])
