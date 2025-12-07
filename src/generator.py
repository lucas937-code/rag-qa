import requests
import torch
from src.config import Config, OllamaConfig, DEFAULT_CONFIG
from src.retriever import Retriever
from src.load_data import load_embeddings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)

# ==============================
# Config
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 48
MAX_INPUT_LENGTH = 2048


# ==============================
# Init models / index (lazy-loaded)
# ==============================

_gen_model = None
_tokenizer = None

def get_generator(model_name: str):
    global _gen_model, _tokenizer

    if _gen_model is None:
        print(f"🔹 Loading generator: {model_name}")

        # Load config and tokenizer
        cfg = AutoConfig.from_pretrained(model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Proper detection: encoder-decoder (T5/BART/etc.) vs decoder-only (GPT/etc.)
        is_encoder_decoder = getattr(cfg, "is_encoder_decoder", False)

        if is_encoder_decoder:
            # Flan-T5, T5, BART, etc. → Seq2Seq
            print("➡ Detected encoder-decoder (Seq2Seq) model.")
            _gen_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE)
        else:
            # GPT-style models → CausalLM
            print("➡ Detected decoder-only (CausalLM) model.")
            if _tokenizer.pad_token_id is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            _gen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
            )

        _gen_model.eval()

    return _tokenizer, _gen_model


# ==============================
# Ollama chat call
# ==============================
def call_ollama(messages, ollama_url, model, max_new_tokens=MAX_GEN_TOKENS):
    resp = requests.post(
        ollama_url,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


# ==============================
# LLM-based evidence selection helpers
# ==============================

def _filter_passages_with_llm_hf(query, passages, config: Config = DEFAULT_CONFIG):
    """
    Use the generator model itself to decide which passages contain enough
    information to answer the question.
    Returns a list of passages predicted as relevant.
    """
    if not passages:
        return []

    tokenizer, model = get_generator(config.generator_model)
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")

    selected = []
    for passage in passages:
        prompt = (
            "You are helping to select useful evidence passages for question answering.\n"
            f"Question: {query}\n\n"
            f"Passage:\n{passage}\n\n"
            "Does this passage contain enough information to answer the question?\n"
            "Answer with a single word: YES or NO."
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
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        if text.startswith("yes"):
            selected.append(passage)

    return selected


def _filter_passages_with_llm_ollama(query, passages, config: OllamaConfig):
    """
    Ollama variant of LLM-based passage filtering.
    """
    if not passages:
        return []

    selected = []
    for passage in passages:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are helping to select useful evidence passages for question answering. "
                    "Decide if the passage contains enough information to answer the question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Passage:\n{passage}\n\n"
                    "Does this passage contain enough information to answer the question?\n"
                    "Answer with a single word: YES or NO."
                ),
            },
        ]

        text = call_ollama(
            messages,
            ollama_url=config.ollama_url,
            model=config.generator_model,
            max_new_tokens=4,
        ).strip().lower()

        if text.startswith("yes"):
            selected.append(passage)

    return selected


def _rewrite_query_with_llm_hf(query, config: Config = DEFAULT_CONFIG):
    """
    Ask the generator to rewrite the question as a concise search query.
    Used for a single optional re-retrieval step when no passage was selected.
    """
    tokenizer, model = get_generator(config.generator_model)
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")

    prompt = (
        "Rewrite the following question as a concise search query containing only the most important keywords.\n"
        "Return only the rewritten query.\n\n"
        f"Question: {query}"
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

    rewritten = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return rewritten or query


def _rewrite_query_with_llm_ollama(query, config: OllamaConfig):
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite user questions into concise search queries that work well for dense passage retrieval. "
                "Keep only the most important keywords."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following question as a concise search query. "
                "Return only the rewritten query.\n\n"
                f"Question: {query}"
            ),
        },
    ]

    rewritten = call_ollama(
        messages,
        ollama_url=config.ollama_url,
        model=config.generator_model,
        max_new_tokens=MAX_GEN_TOKENS,
    ).strip()

    return rewritten or query


# ==============================
# Answer-in-context helper
# ==============================

def _normalize_text_for_match(text: str) -> str:
    text = text.lower().strip()
    # collapse internal whitespace
    return " ".join(text.split())


def _answer_in_passages(answer: str, passages) -> bool:
    if not answer:
        return False
    norm_ans = _normalize_text_for_match(answer)
    if not norm_ans:
        return False
    for p in passages:
        if norm_ans in _normalize_text_for_match(p):
            return True
    return False


# ==============================
# Generation helpers (HF / Ollama)
# ==============================

def _generate_from_passages_hf(query, top_passages, config: Config, allow_idk: bool):
    tokenizer, model = get_generator(config.generator_model)
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")

    context_block = "\n\n".join(
        [f"Passage {i+1}:\n{p}" for i, p in enumerate(top_passages)]
    )

    base_instructions = (
        "You are a question answering assistant for a trivia dataset. "
        "Use only the information in the passages to answer the question. "
        "Prefer short, factual answers. "
        "Your answer must be copied exactly from the passages, "
        "or be a very small normalization (for example, different casing or removing 'the'). "
    )

    if allow_idk:
        idk_instruction = (
            "If no word, phrase, name, or number in the passages could reasonably answer the question, "
            "respond exactly with \"I don't know\". "
        )
    else:
        idk_instruction = (
            "Always choose the most plausible answer mentioned in the passages. "
            "Never respond with \"I don't know\". "
        )

    prompt = (
        base_instructions
        + idk_instruction
        + "\n\n"
        f"Question: {query}\n\n"
        f"Passages:\n{context_block}\n\n"
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


def _generate_from_passages_ollama(query, top_passages, config: OllamaConfig, allow_idk: bool):
    context_block = "\n\n".join(
        [f"Passage {i+1}:\n{p}" for i, p in enumerate(top_passages)]
    )

    base_system = (
        "You are a question answering assistant for a trivia dataset. "
        "Use only the information in the passages to answer the question. "
        "Prefer short, factual answers. "
        "Your answer must be copied exactly from the passages, "
        "or be a very small normalization (for example, different casing or removing 'the'). "
    )

    if allow_idk:
        idk_system = (
            "If no word, phrase, name, or number in the passages could reasonably answer the question, "
            "respond exactly with \"I don't know\". "
        )
    else:
        idk_system = (
            "Always choose the most plausible answer mentioned in the passages. "
            "Never respond with \"I don't know\". "
        )

    messages = [
        {
            "role": "system",
            "content": base_system + idk_system,
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Passages:\n{context_block}\n\n"
                "Give only the final answer as a short phrase (no explanation, no full sentence)."
            ),
        },
    ]

    answer = call_ollama(
        messages,
        ollama_url=config.ollama_url,
        model=config.generator_model,
        max_new_tokens=MAX_GEN_TOKENS,
    ).strip()
    return answer


# ==============================
# Generate answer with LLM-selected evidence
# ==============================
def generate_answer_combined_ollama(query, retriever, corpus, embeddings, config: OllamaConfig, top_k=5):
    # First retrieve a larger candidate set
    candidate_k = max(top_k * 4, top_k + 5)
    candidate_passages, _ = retriever.retrieve_top_k(
        query,
        corpus,
        embeddings,
        config.embedding_model,
        config.rerank_model,
        k=candidate_k,
    )

    # LLM-based filtering of candidates
    selected_passages = _filter_passages_with_llm_ollama(query, candidate_passages, config=config)

    # Optional: single re-retrieval with rewritten query if nothing was selected
    if not selected_passages:
        rewritten_query = _rewrite_query_with_llm_ollama(query, config=config)
        retry_candidates, _ = retriever.retrieve_top_k(
            rewritten_query,
            corpus,
            embeddings,
            config.embedding_model,
            config.rerank_model,
            k=candidate_k,
        )
        selected_passages = _filter_passages_with_llm_ollama(rewritten_query, retry_candidates, config=config)

    # Fallback: if still nothing selected, fall back to top_k candidates from the first retrieval
    if not selected_passages:
        top_passages = candidate_passages[:top_k]
    else:
        top_passages = selected_passages[:top_k]

    # First pass: allow "I don't know"
    answer = _generate_from_passages_ollama(query, top_passages, config=config, allow_idk=True)
    norm_answer = answer.lower().strip()

    # If model says "I don't know", try a second pass where it must pick a candidate
    if norm_answer in {"i don't know", "i dont know", "idk"}:
        second_answer = _generate_from_passages_ollama(query, top_passages, config=config, allow_idk=False)
        if _answer_in_passages(second_answer, top_passages):
            answer = second_answer
        # else keep original "I don't know"
    else:
        # If answer is not found in the passages, be conservative and fall back to "I don't know"
        if not _answer_in_passages(answer, top_passages):
            answer = "I don't know"

    return answer, top_passages

def generate_answer_combined_hf(query, retriever, corpus, embeddings, top_k=5, config: Config = DEFAULT_CONFIG):
    # First retrieve a larger candidate set
    candidate_k = max(top_k * 4, top_k + 5)
    candidate_passages, _ = retriever.retrieve_top_k(
        query,
        corpus,
        embeddings,
        config.embedding_model,
        config.rerank_model,
        k=candidate_k,
    )

    # LLM-based filtering of candidates
    selected_passages = _filter_passages_with_llm_hf(query, candidate_passages, config=config)

    # Optional: single re-retrieval with rewritten query if nothing was selected
    if not selected_passages:
        rewritten_query = _rewrite_query_with_llm_hf(query, config=config)
        retry_candidates, _ = retriever.retrieve_top_k(
            rewritten_query,
            corpus,
            embeddings,
            config.embedding_model,
            config.rerank_model,
            k=candidate_k,
        )
        selected_passages = _filter_passages_with_llm_hf(rewritten_query, retry_candidates, config=config)

    # Fallback: if still nothing selected, fall back to top_k candidates from the first retrieval
    if not selected_passages:
        top_passages = candidate_passages[:top_k]
    else:
        top_passages = selected_passages[:top_k]

    # First pass: allow "I don't know"
    answer = _generate_from_passages_hf(query, top_passages, config=config, allow_idk=True)
    norm_answer = answer.lower().strip()

    # If model says "I don't know", try a second pass where it must pick a candidate
    if norm_answer in {"i don't know", "i dont know", "idk"}:
        second_answer = _generate_from_passages_hf(query, top_passages, config=config, allow_idk=False)
        if _answer_in_passages(second_answer, top_passages):
            answer = second_answer
        # else keep original "I don't know"
    else:
        # If answer is not found in the passages, be conservative and fall back to "I don't know"
        if not _answer_in_passages(answer, top_passages):
            answer = "I don't know"

    return answer, top_passages

def generate_answer_combined(query, retriever, corpus, embeddings, top_k=5, config: Config = DEFAULT_CONFIG):
    if isinstance(config, OllamaConfig):
        return generate_answer_combined_ollama(query, retriever, corpus, embeddings, config=config, top_k=top_k)
    else:
        return generate_answer_combined_hf(query, retriever, corpus, embeddings, top_k=top_k, config=config)

# ==============================
# Manual test (cmd terminal)
# ==============================
if __name__ == "__main__":
    corpus, emb = load_embeddings()
    q = "The medical condition glaucoma affects which part of the body?"

    retriever = Retriever()
    passages, _ = retriever.retrieve_top_k(q, corpus, emb, DEFAULT_CONFIG.embedding_model, DEFAULT_CONFIG.rerank_model, k=3)
    print("\nRetrieved passages:")
    for i, p in enumerate(passages):
        print(f"{i+1}. {p[:200].replace(chr(10),' ')}...")

    print("\nGenerated Answer:")
    print(generate_answer_combined(q, corpus, retriever, emb, top_k=5)[0])
