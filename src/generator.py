import requests
import torch
from src.config import Config, OllamaConfig, LocalConfig, ColabConfig
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)

class Generator(ABC):
    def __init__(self, 
                 config: Config,
                 max_gen_tokens: int = 48,
                 max_input_length: int = 2048) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_gen_tokens = max_gen_tokens
        self.max_input_length = max_input_length
        self.model_name = config.generator_model

    @abstractmethod
    def generate(self, query: str, passages: list):
        pass
    
class HFGenerator(Generator):
    def __init__(self, 
                 config: LocalConfig | ColabConfig,
                 max_gen_tokens: int = 48,
                 max_input_length: int = 2048):
        super().__init__(config, max_gen_tokens, max_input_length)

    def _get_generator(self, gen_model_name: str, device: str):
        cfg = AutoConfig.from_pretrained(gen_model_name, use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)
        
        is_concoder_decoder = getattr(cfg, "is_encoder_decoder", False)

        if is_concoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                gen_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
        else:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                gen_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
        return tokenizer, model

    def generate(self, query: str, passages: list):
        tokenizer, gen_model = self._get_generator(self.model_name, self.device)
        if tokenizer is None or gen_model is None:
            raise ValueError("HuggingFace tokenizer or model is not initialized.")
        
        context_block = "\n---\n".join(passages)
        prompt = (
            "You are a concise QA assistant. Answer ONLY from the provided context. "
            "If you are unsure, reply \"I don't know\". Respond in <=20 words.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
            padding=True).to(gen_model.device)
        
        with torch.no_grad():
            output = gen_model.generate(
                **inputs,
                max_new_tokens=self.max_gen_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id)
            
        answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return answer
    
class OllamaGenerator(Generator):
    def __init__(self, 
                 config: OllamaConfig,
                 max_gen_tokens: int = 48,
                 max_input_length: int = 2048):
        super().__init__(config, max_gen_tokens, max_input_length)
        self.ollama_url = config.ollama_url

    def _call_ollama(self, messages: list, max_gen_tokens: int):
        if self.ollama_url is None:
            raise ValueError("Ollama URL is not set in the configuration.")
        
        resp = requests.post(
        self.ollama_url,
        json={
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_gen_tokens},
        },
        timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    def generate(self, query: str, passages: list):
        context_block = "\n---\n".join(passages)

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
        
        answer = self._call_ollama(messages, self.max_gen_tokens)
        return answer