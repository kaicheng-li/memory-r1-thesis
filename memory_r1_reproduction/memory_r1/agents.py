from __future__ import annotations

import json
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .parser import parse_answer_output, parse_fact_output, parse_manager_output
from .prompts import build_answer_prompt, build_fact_extraction_prompt, build_manager_prompt
from .schemas import AnswerTrace, DialogueTurn, MemoryDecision, MemoryEntry


class HFTextGenerator:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: Optional[bool] = None,
        num_return_sequences: int = 1,
    ) -> List[str]:
        do_sample = do_sample if do_sample is not None else temperature > 0
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        texts = []
        for output in outputs:
            texts.append(self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip())
        return texts


class FactExtractor:
    def __init__(self, generator: Optional[HFTextGenerator] = None):
        self.generator = generator

    def extract(self, turn: DialogueTurn) -> List[str]:
        if self.generator is None:
            return self._heuristic_extract(turn)
        prompt = build_fact_extraction_prompt(turn)
        output = self.generator.generate(prompt, max_new_tokens=128, temperature=0.0)[0]
        facts = parse_fact_output(output)
        return facts or self._heuristic_extract(turn)

    def _heuristic_extract(self, turn: DialogueTurn) -> List[str]:
        text = turn.text.strip()
        if not text:
            return []
        chunks = [chunk.strip(" .") for chunk in text.replace(";", ".").split(".") if chunk.strip()]
        return [f"{turn.speaker}: {chunk}" for chunk in chunks[:3]]


class MemoryManagerAgent:
    def __init__(self, generator: HFTextGenerator):
        self.generator = generator

    def propose(
        self,
        turn: DialogueTurn,
        facts: List[str],
        retrieved_memories: List[MemoryEntry],
        temperature: float = 0.0,
    ) -> List[MemoryDecision]:
        prompt = build_manager_prompt(turn, facts, retrieved_memories)
        text = self.generator.generate(prompt, max_new_tokens=256, temperature=temperature)[0]
        return parse_manager_output(text)


class AnswerAgent:
    def __init__(self, generator: HFTextGenerator):
        self.generator = generator

    def answer(
        self,
        question: str,
        retrieved_memories: List[MemoryEntry],
        temperature: float = 0.0,
    ) -> AnswerTrace:
        prompt = build_answer_prompt(question, retrieved_memories)
        text = self.generator.generate(prompt, max_new_tokens=256, temperature=temperature)[0]
        return parse_answer_output(text)

    @staticmethod
    def build_supervision_target(selected_memory_ids: List[str], answer: str) -> str:
        return json.dumps(
            {
                "selected_memory_ids": selected_memory_ids,
                "answer": answer,
            },
            ensure_ascii=False,
        )
