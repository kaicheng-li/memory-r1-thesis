"""Algorithm 3: construct a global memory bank with the Memory Manager."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_extractor import build_extract_input
from memfactory.modules.memory_updater import build_manager_input


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _session_number(name: str) -> int:
    match = re.search(r"session_(\d+)$", name)
    return int(match.group(1)) if match else 10**9


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if "turns" in sample:
        return sample
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo conversation must be an object.")
    turns = []
    session_names = sorted(
        (name for name, value in conversation.items() if re.fullmatch(r"session_\d+", name) and isinstance(value, list)),
        key=_session_number,
    )
    for session_name in session_names:
        for local_index, raw_turn in enumerate(conversation[session_name]):
            if not isinstance(raw_turn, dict):
                continue
            turns.append(
                {
                    "speaker": str(raw_turn.get("speaker", "")),
                    "text": str(raw_turn.get("text", "")),
                    "timestamp": str(raw_turn.get("timestamp", conversation.get(f"{session_name}_date_time", ""))),
                    "dia_id": str(raw_turn.get("dia_id", f"D{_session_number(session_name)}:{local_index}")),
                }
            )
    return {
        "dialogue_id": str(sample.get("sample_id", sample.get("dialogue_id", "dialogue"))),
        "participants": [str(value) for key in ("speaker_a", "speaker_b") if (value := conversation.get(key))],
        "turns": turns,
    }


def load_dialogues(payload: Any) -> list[dict[str, Any]]:
    samples = payload if isinstance(payload, list) else [payload]
    return [normalize_sample(sample) for sample in samples if isinstance(sample, dict)]


def score(query: str, memory: dict[str, Any]) -> float:
    query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    memory_tokens = set(re.findall(r"[a-zA-Z0-9]+", str(memory.get("text", "")).lower()))
    denominator = math.sqrt(len(query_tokens) * len(memory_tokens))
    return len(query_tokens & memory_tokens) / denominator if denominator else 0.0


def top_k(query: str, memory_bank: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(memory_bank, key=lambda memory: score(query, memory), reverse=True)[:k]


def top_k_per_speaker(
    query: str,
    memory_bank: list[dict[str, Any]],
    participants: list[str],
    k: int,
) -> list[dict[str, Any]]:
    return [
        memory
        for participant in participants
        for memory in top_k(
            query,
            [entry for entry in memory_bank if entry["speaker"] == participant],
            k,
        )
    ]


def parse_manager_output(text: str) -> list[dict[str, Any]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Memory Manager output does not contain JSON.")
    payload = json.loads(text[start : end + 1])
    decisions = payload.get("memory", [])
    if not isinstance(decisions, list):
        raise ValueError("Memory Manager output has no memory list.")
    return [item for item in decisions if isinstance(item, dict)]


def apply_decisions(
    memory_bank: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    dialogue_id: str,
    turn_index: int,
    default_speaker: str,
    timestamp: str,
) -> None:
    by_id = {str(memory.get("id", "")): memory for memory in memory_bank}
    next_id = len(memory_bank)
    for decision in decisions:
        event = str(decision.get("event", "NONE")).upper()
        memory_id = str(decision.get("id", ""))
        text = str(decision.get("text", "")).strip()
        if event == "ADD" and text:
            entry = {
                "id": f"{dialogue_id}:m{next_id}",
                "text": text,
                "source_turn": turn_index,
                "speaker": str(decision.get("speaker", default_speaker)),
                "timestamp": str(decision.get("timestamp", timestamp)),
            }
            next_id += 1
            memory_bank.append(entry)
            by_id[entry["id"]] = entry
        elif event == "UPDATE" and memory_id in by_id and text:
            by_id[memory_id]["text"] = text
            by_id[memory_id]["source_turn"] = turn_index
        elif event == "DELETE" and memory_id in by_id:
            entry = by_id.pop(memory_id)
            memory_bank.remove(entry)


class Manager:
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

    def extract(self, turn: dict[str, Any]) -> list[dict[str, Any]]:
        """LLMExtract(di) at deployment: the trained manager extracts the turn's facts."""
        prompt = build_extract_input(turn)
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
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            return []
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
        raw_facts = payload.get("memory_list", [])
        if not isinstance(raw_facts, list):
            return []
        return [
            {
                "speaker": str(turn.get("speaker", "")),
                "timestamp": str(turn.get("timestamp", "")),
                "key": str(fact.get("key", "")),
                "memory_type": str(fact.get("memory_type", "")),
                "tags": fact.get("tags", []),
                "text": str(fact.get("value", "")).strip(),
            }
            for fact in raw_facts
            if isinstance(fact, dict) and str(fact.get("value", "")).strip()
        ]

    def generate(self, old_memory: list[dict[str, Any]], facts: list[dict[str, Any]], top_k: int) -> str:
        query = " ".join(fact["text"] for fact in facts)
        retrieved = top_k_fn(query, old_memory, top_k)
        prompt = build_manager_input(retrieved, facts)
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
        return self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()


top_k_fn = top_k


def construct(input_path: str, output_path: str, model_path: str, device: str, retrieval_top_k: int, max_new_tokens: int) -> None:
    manager = Manager(model_path, device, max_new_tokens)
    samples = load_dialogues(read_json(input_path))
    results = []
    for sample in samples:
        dialogue_id = str(sample.get("dialogue_id", "dialogue"))
        memory_bank = []
        for turn_index, turn in enumerate(sample.get("turns", [])):
            facts = manager.extract(turn)
            if not facts:
                continue
            output = manager.generate(memory_bank, facts, retrieval_top_k)
            apply_decisions(
                memory_bank,
                parse_manager_output(output),
                dialogue_id,
                turn_index,
                str(turn.get("speaker", "")),
                str(turn.get("timestamp", "")),
            )
        results.append({
            "dialogue_id": dialogue_id,
            "participants": sample.get("participants", []),
            "memory_bank": memory_bank,
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} memory banks to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 3: construct Memory-R1 memory banks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    construct(args.input, args.output, args.manager_model, args.device, args.retrieval_top_k, args.max_new_tokens)


if __name__ == "__main__":
    main()
