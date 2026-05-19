from ..utils import build_final_answer_messages, call_label, call_llm_api


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
    messages = build_final_answer_messages(
        messages=parameters["messages"],
        question=question,
        user_prefix="You have reached the final answer stage.",
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
