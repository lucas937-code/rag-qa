from src.generator import call_ollama, MAX_GEN_TOKENS, MAX_INPUT_LENGTH, get_generator
from src.config import OllamaConfig, Config
import torch

def generate_answer_ollama(query, config: OllamaConfig):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a question answering assistant for a trivia dataset. "
                "Prefer short, factual answers. "
                "If you are not able to answer the question, respond exactly with \"I don't know\"."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                "Give only the final answer as a short phrase (no explanation, no full sentence)."
            ),
        },
    ]

    answer = call_ollama(messages, config.ollama_url, model=config.generator_model, max_new_tokens=MAX_GEN_TOKENS).strip()
    return answer

def generate_answer_hf(query, config: Config):
    tokenizer, model = get_generator(config.generator_model)
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")

    prompt = (
        "You are a question answering assistant for a trivia dataset. "
        "Prefer short, factual answers. "
        "If you are not able to answer the question, respond exactly with \"I don't know\".\n\n"
        f"Question: {query}\n\n"
        "Give only the final answer as a short phrase (no explanation, no full sentence)."
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer

def generate_answer_no_rag(query, config: Config):
    if isinstance(config, OllamaConfig):
        return generate_answer_ollama(query, config)
    else:
        return generate_answer_hf(query, config)