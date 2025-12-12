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
def call_ollama(
    messages,
    ollama_url,
    model,
    max_new_tokens=MAX_GEN_TOKENS,
    temperature=None,
    top_p=None,
):
    options = {"num_predict": max_new_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p

    resp = requests.post(
        ollama_url,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


# ==============================
# Generate answer from combined top-K context
# ==============================
def generate_answer_combined_ollama(query, retriever, corpus, embeddings, config: OllamaConfig, top_k=5):
    top_passages, _ = retriever.retrieve_top_k(query, corpus, embeddings, config.embedding_model, config.rerank_model, k=top_k)
    context_block = "\n\n".join(
        [f"Passage {i+1}:\n{p}" for i, p in enumerate(top_passages)]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a question answering assistant for a trivia dataset. "
                "Use the information in the passages to answer the question. "
                "Prefer short, factual answers. "
                "If the passages do not clearly contain the answer, answer it based on your own knowledge. "
            ),
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

    answer = call_ollama(messages, config.ollama_url, model=config.generator_model, max_new_tokens=MAX_GEN_TOKENS).strip()
    return answer, top_passages

def generate_answer_combined_hf(query, retriever, corpus, embeddings, top_k=5, config: Config = DEFAULT_CONFIG):
    tokenizer, model = get_generator(config.generator_model)
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")

    top_passages, _ = retriever.retrieve_top_k(query, corpus, embeddings, config.embedding_model, config.rerank_model, k=top_k)
    context_block = "\n\n".join(
        [f"Passage {i+1}:\n{p}" for i, p in enumerate(top_passages)]
    )

    prompt = (
        "You are a question answering assistant for a trivia dataset. "
        "Use the information in the passages to answer the question. "
        "Prefer short, factual answers. "
        "If the passages do not clearly contain the answer, answer it based on your own knowledge. \n\n"
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
    return answer, top_passages

def generate_answer_combined(query, retriever, corpus, embeddings, top_k=5, config: Config = DEFAULT_CONFIG):
    if isinstance(config, OllamaConfig):
        return generate_answer_combined_ollama(query, retriever, corpus, embeddings, config=config, top_k=top_k)
    else:
        return generate_answer_combined_hf(query, retriever, corpus, embeddings, top_k=top_k, config=config)


# ==============================
# Self-consistency consensus generation
# ==============================

def _normalize_for_voting(text: str) -> str:
    text = text.strip().strip("\"'")
    text = text.lower()
    # strip trailing punctuation
    while text and text[-1] in ".!?,":
        text = text[:-1]
    # remove leading articles
    for art in ("the ", "a ", "an "):
        if text.startswith(art):
            text = text[len(art) :]
            break
    # collapse whitespace
    return " ".join(text.split())


def _answer_in_passages(answer_norm: str, passages) -> bool:
    if not answer_norm:
        return False
    for p in passages:
        p_norm = " ".join(p.lower().split())
        if answer_norm in p_norm:
            return True
    return False


def _generate_single_hf(
    prompt: str,
    tokenizer,
    model,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": MAX_GEN_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def generate_answer_consensus_hf(
    query,
    retriever,
    corpus,
    embeddings,
    top_k=5,
    config: Config = DEFAULT_CONFIG,
    n_samples: int = 3,
):
    tokenizer, model = get_generator(config.generator_model)
    if tokenizer is None or model is None:
        raise RuntimeError("Generator model or tokenizer not loaded.")

    if n_samples < 1:
        n_samples = 1

    top_passages, _ = retriever.retrieve_top_k(
        query,
        corpus,
        embeddings,
        config.embedding_model,
        config.rerank_model,
        k=top_k,
    )
    context_block = "\n\n".join(
        [f"Passage {i+1}:\n{p}" for i, p in enumerate(top_passages)]
    )

    prompt = (
        "You are a question answering assistant for a trivia dataset. "
        "Use the information in the passages to answer the question. "
        "Prefer short, factual answers. "
        "If the passages do not clearly contain the answer, you may answer based on your own knowledge.\n\n"
        f"Question: {query}\n\n"
        f"Passages:\n{context_block}\n\n"
        "Give only the final answer as a short phrase (no explanation, no full sentence)."
    )

    # Deterministic baseline answer
    deterministic_answer = _generate_single_hf(
        prompt, tokenizer, model, do_sample=False
    )

    answers = [deterministic_answer]
    # Additional stochastic samples
    for _ in range(max(n_samples - 1, 0)):
        ans = _generate_single_hf(
            prompt,
            tokenizer,
            model,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        answers.append(ans)

    # Voting with context check
    counts = {}
    repr_map = {}
    in_context = set()
    for ans in answers:
        norm = _normalize_for_voting(ans)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        if norm not in repr_map:
            repr_map[norm] = ans
        if _answer_in_passages(norm, top_passages):
            in_context.add(norm)

    if in_context:
        best_norm = max(in_context, key=lambda n: counts.get(n, 0))
        consensus_answer = repr_map[best_norm]
    else:
        consensus_answer = deterministic_answer

    return consensus_answer, top_passages


def generate_answer_consensus_ollama(
    query,
    retriever,
    corpus,
    embeddings,
    config: OllamaConfig,
    top_k=5,
    n_samples: int = 3,
):
    if n_samples < 1:
        n_samples = 1

    top_passages, _ = retriever.retrieve_top_k(
        query,
        corpus,
        embeddings,
        config.embedding_model,
        config.rerank_model,
        k=top_k,
    )
    context_block = "\n\n".join(
        [f"Passage {i+1}:\n{p}" for i, p in enumerate(top_passages)]
    )

    def _ollama_call(do_sample: bool):
        temperature = 0.7 if do_sample else None
        top_p = 0.9 if do_sample else None
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a question answering assistant for a trivia dataset. "
                    "Use the information in the passages to answer the question. "
                    "Prefer short, factual answers. "
                    "If the passages do not clearly contain the answer, you may answer based on your own knowledge."
                ),
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
        return call_ollama(
            messages,
            ollama_url=config.ollama_url,
            model=config.generator_model,
            max_new_tokens=MAX_GEN_TOKENS,
            temperature=temperature,
            top_p=top_p,
        ).strip()

    deterministic_answer = _ollama_call(do_sample=False)

    answers = [deterministic_answer]
    for _ in range(max(n_samples - 1, 0)):
        ans = _ollama_call(do_sample=True)
        answers.append(ans)

    counts = {}
    repr_map = {}
    in_context = set()
    for ans in answers:
        norm = _normalize_for_voting(ans)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        if norm not in repr_map:
            repr_map[norm] = ans
        if _answer_in_passages(norm, top_passages):
            in_context.add(norm)

    if in_context:
        best_norm = max(in_context, key=lambda n: counts.get(n, 0))
        consensus_answer = repr_map[best_norm]
    else:
        consensus_answer = deterministic_answer

    return consensus_answer, top_passages


def generate_answer_consensus(
    query,
    retriever,
    corpus,
    embeddings,
    top_k=5,
    config: Config = DEFAULT_CONFIG,
    n_samples: int = 3,
):
    if isinstance(config, OllamaConfig):
        return generate_answer_consensus_ollama(
            query,
            retriever,
            corpus,
            embeddings,
            config=config,
            top_k=top_k,
            n_samples=n_samples,
        )
    else:
        return generate_answer_consensus_hf(
            query,
            retriever,
            corpus,
            embeddings,
            top_k=top_k,
            config=config,
            n_samples=n_samples,
        )

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
