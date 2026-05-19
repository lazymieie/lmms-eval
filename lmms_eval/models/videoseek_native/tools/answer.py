from ..utils import call_label, call_llm_api


answer_tool = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Based on the given trajectory, generate the final answer to the question.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}


def execute_answer(config: dict, parameters: dict) -> str:
    question = parameters["question"]
    messages = list(parameters["messages"])
    messages.append(
        {
            "role": "system",
            "content": (
                "You are now in the final answer stage. "
                "Do not call any tool. "
                "Do not output XML, JSON, code fences, or any tool-call syntax. "
                "Ignore earlier instructions that asked for tool calls. "
                "Answer directly using the requested final answer format only."
            ),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Provide the final answer now. "
                "Do not call any tool and do not propose further actions. "
                "If this is a multiple-choice question, respond with only the single option letter from the given choices."
            ),
        }
    )
    with call_label("answer"):
        response = call_llm_api(
            messages=messages,
            model_name=config["model_name"],
            api_base=config["api_base"],
            api_key=config["api_key"],
            api_version=config["api_version"],
            max_tokens=config["max_tokens"],
            reasoning_effort=config["reasoning_effort"],
            seed=config["seed"],
            temperature=config["temperature"],
            timeout=config.get("timeout", 900),
        )
    return response.choices[0].message.content
