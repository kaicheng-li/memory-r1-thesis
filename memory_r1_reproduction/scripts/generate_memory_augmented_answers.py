"""Algorithm 4: generate answers from a constructed global memory bank."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_retriever import build_answer_input


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def score(query: str, memory: dict[str, Any]) -> float:
    query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    memory_tokens = set(re.findall(r"[a-zA-Z0-9]+", str(memory.get("text", "")).lower()))
    denominator = math.sqrt(len(query_tokens) * len(memory_tokens))
    return len(query_tokens & memory_tokens) / denominator if denominator else 0.0


def top_k(query: str, memory_bank: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(memory_bank, key=lambda memory: score(query, memory), reverse=True)[:k]


def raw_questions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    questions = sample.get("questions", sample.get("qa", []))
    result = []
    for index, item in enumerate(questions):
        if not isinstance(item, dict):
            continue
        question = item.get("question", item.get("query", ""))
        if not question:
            continue
        result.append({
            "question_id": str(item.get("question_id", index)),
            "question": str(question),
            "answer": str(item.get("answer", item.get("gold_answer", ""))),
        })
    return result


class AnswerAgent:
    def __init__(self, model_path: str, device: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def answer(self, question: str, memories: list[dict[str, Any]]) -> tuple[str, str]:
        prompt = build_answer_input(question, {"Memory Bank": memories})
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        raw = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
        answer = raw.rsplit("Answer:", 1)[-1].strip()
        return answer, raw


def generate(input_path: str, memory_bank_path: str, output_path: str, answer_model: str, device: str, retrieval_top_k: int, max_new_tokens: int) -> None:
    samples = read_json(input_path)
    samples = samples if isinstance(samples, list) else [samples]
    memory_banks = {str(item["dialogue_id"]): item.get("memory_bank", []) for item in read_json(memory_bank_path)}
    agent = AnswerAgent(answer_model, device, max_new_tokens)
    rows = []
    for sample in samples:
        dialogue_id = str(sample.get("sample_id", sample.get("dialogue_id", "dialogue")))
        bank = memory_banks.get(dialogue_id, [])
        for question in raw_questions(sample):
            retrieved = top_k(question["question"], bank, retrieval_top_k)
            answer, raw_output = agent.answer(question["question"], retrieved)
            rows.append({
                "dialogue_id": dialogue_id,
                "question_id": question["question_id"],
                "question": question["question"],
                "retrieved_memories": retrieved,
                "gold_answer": question["answer"],
                "answer": answer,
                "raw_output": raw_output,
            })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} answers to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 4: generate memory-augmented answers.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--memory-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval-top-k", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    generate(args.input, args.memory_bank, args.output, args.answer_model, args.device, args.retrieval_top_k, args.max_new_tokens)


if __name__ == "__main__":
    main()
