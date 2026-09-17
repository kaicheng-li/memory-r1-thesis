"""Algorithm 3: construct a global memory bank with the Memory Manager."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_extractor import build_extract_input
from memfactory.modules.memory_updater import build_manager_input
from memfactory.memory_runtime import apply_decisions, parse_manager_output, retrieve


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
        retrieved = retrieve(query, old_memory, top_k)
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
                turn,
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
