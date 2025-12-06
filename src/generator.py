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

class Generator:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_gen_tokens = 48
        self.max_input_length = 2048
        self.gen_model = None
        self.tokenizer = None
        self.retriever = Retriever()

    def get_generator(self, model_name: str):
        if self.gen_model is None:
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
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
            else:
                # GPT-style models → CausalLM
                print("➡ Detected decoder-only (CausalLM) model.")
                if _tokenizer.pad_token_id is None:
                    _tokenizer.pad_token = _tokenizer.eos_token
                _gen_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                )

            _gen_model.eval()

        return self.tokenizer, self.gen_model    
    
    def call_ollama(self, messages, ollama_url, model, max_new_tokens):
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
    
    def generate_answer_combined_ollama(self, query, corpus, embeddings, config: OllamaConfig, top_k=5):
        top_passages, _ = self.retriever.retrieve_top_k(query, corpus, embeddings, config.embedding_model, config.rerank_model, k=top_k)
        context_block = "\n---\n".join(top_passages)

        # Simple text prompt for Flan-T5 (no chat template)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise QA assistant. Answer ONLY from the provided context. "
                    "If you are unsure, reply \"I don't know\". Respond with the answer only."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:",
            },
        ]

        answer = self.call_ollama(messages, config.ollama_url, model=config.generator_model, max_new_tokens=self.max_gen_tokens).strip()
        return answer, top_passages

    def generate_answer_combined_hf(self, query, corpus, embeddings, top_k=5, config: Config = DEFAULT_CONFIG):
        tokenizer, model = self.get_generator(config.generator_model)
        if tokenizer is None or model is None:
            raise RuntimeError("Generator model or tokenizer not loaded.")

        top_passages, _ = self.retriever.retrieve_top_k(query, corpus, embeddings, config.embedding_model, config.rerank_model, k=top_k)
        context_block = "\n---\n".join(top_passages)

        # Simple text prompt for Flan-T5 (no chat template)
        prompt = (
            "You are a concise QA assistant. Answer ONLY from the provided context. "
            "If you are unsure, reply \"I don't know\". Respond in <=20 words.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_gen_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return answer, top_passages

    def generate_answer_combined(self, query, corpus, embeddings, top_k=5, config: Config = DEFAULT_CONFIG):
        if isinstance(config, OllamaConfig):
            return self.generate_answer_combined_ollama(query, corpus, embeddings, config=config, top_k=top_k)
        else:
            return self.generate_answer_combined_hf(query, corpus, embeddings, top_k=top_k, config=config)

# ==============================
# Manual test (cmd terminal)
# ==============================
if __name__ == "__main__":
    corpus, emb = load_embeddings()
    q = "The medical condition glaucoma affects which part of the body?"

    retriever = Retriever()
    generator = Generator()
    passages, _ = retriever.retrieve_top_k(q, corpus, emb, DEFAULT_CONFIG.embedding_model, DEFAULT_CONFIG.rerank_model, k=3)
    print("\nRetrieved passages:")
    for i, p in enumerate(passages):
        print(f"{i+1}. {p[:200].replace(chr(10),' ')}...")

    print("\nGenerated Answer:")
    print(generator.generate_answer_combined(q, corpus, emb, top_k=5)[0])