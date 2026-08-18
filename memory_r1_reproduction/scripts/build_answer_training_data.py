"""Build Answer Agent training tuples from raw LoCoMo JSON.

The manager is replayed over every turn in each dialogue. For every question,
the script retrieves the 60 most relevant memories from the global memory bank.
"""

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

from memfactory.modules.memory_updater import build_manager_input

TOP_K = 60

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _session_number(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def normalize_locomo_sample(sample: dict[str, Any]) -> dict[str, Any]:
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo sample conversation must be an object.")
    turns = []
    session_names = sorted(
        (name for name, value in conversation.items() if name.startswith("session_") and isinstance(value, list)),
        key=_session_number,
    )
    evidence_to_index = {}
    for session_name in session_names:
        for local_index, raw_turn in enumerate(conversation[session_name]):
            if not isinstance(raw_turn, dict):
                continue
            turn_index = len(turns)
            dia_id = str(raw_turn.get("dia_id", f"D{_session_number(session_name)}:{local_index}"))
            turns.append({
                "speaker": str(raw_turn.get("speaker", "")),
                "text": str(raw_turn.get("text", "")),
                "timestamp": str(raw_turn.get("timestamp", conversation.get(f"{session_name}_date_time", ""))),
                "dia_id": dia_id,
            })
            evidence_to_index[dia_id] = turn_index

    questions = []
    for qa_index, raw_qa in enumerate(sample.get("qa", [])):
        if not isinstance(raw_qa, dict):
            continue
        evidence = raw_qa.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        turn_index = raw_qa.get("turn_index")
        if turn_index is None:
            turn_index = next((evidence_to_index.get(str(item)) for item in evidence if str(item) in evidence_to_index), None)
        if turn_index is None:
            continue
        answer = raw_qa.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(item) for item in answer)
        questions.append({
            "question_id": str(raw_qa.get("question_id", qa_index)),
            "turn_index": int(turn_index),
            "question": str(raw_qa.get("question", "")),
            "answer": str(answer),
            "evidence": evidence,
        })
    return {
        "dialogue_id": str(sample.get("sample_id", sample.get("dialogue_id", "dialogue"))),
        "turns": turns,
        "questions": questions,
    }


def load_dialogues(payload: Any) -> list[dict[str, Any]]:
    samples = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("Input must be raw LoCoMo JSON object or list of objects.")
    return [normalize_locomo_sample(sample) if "conversation" in sample else sample for sample in samples]


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Manager output does not contain a JSON object.")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Manager output is not a JSON object.")
    return payload


class ManagerRunner:
    def __init__(self, model_path: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to replay the trained Memory Manager.") from exc

        self.torch = torch
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()


def manager_prompt(current_turn: dict[str, Any], memory_bank: list[dict[str, Any]]) -> str:
    existing = [
        {
            "id": memory.get("id", ""),
            "speaker": memory.get("speaker", ""),
            "text": memory.get("text", ""),
            "timestamp": memory.get("timestamp", ""),
        }
        for memory in memory_bank
    ]
    return build_manager_input(existing, [current_turn])


def apply_manager_output(memory_bank: list[dict[str, Any]], output: str, dialogue_id: str, turn_index: int) -> None:
    payload = parse_json(output)
    decisions = payload.get("memory", [])
    if not isinstance(decisions, list):
        raise ValueError(f"Invalid manager memory list at {dialogue_id} turn {turn_index}.")

    by_id = {str(memory.get("id", "")): memory for memory in memory_bank}
    next_id = len(memory_bank)
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        event = str(decision.get("event", "NONE")).upper()
        memory_id = str(decision.get("id", ""))
        text = str(decision.get("text", "")).strip()
        if event == "ADD":
            if not text:
                continue
            memory = {
                "id": f"{dialogue_id}:m{next_id}",
                    "text": text,
                "timestamp": str(decision.get("timestamp", "")),
                "source_turn": turn_index,
            }
            next_id += 1
            memory_bank.append(memory)
            by_id[memory["id"]] = memory
        elif event == "UPDATE" and memory_id in by_id and text:
            by_id[memory_id]["text"] = text
            by_id[memory_id]["source_turn"] = turn_index
        elif event == "DELETE" and memory_id in by_id:
            memory_bank.remove(by_id[memory_id])
            del by_id[memory_id]


def word_score(question: str, memory: dict[str, Any]) -> float:
    question_tokens = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    memory_tokens = set(re.findall(r"[a-zA-Z0-9]+", memory.get("text", "").lower()))
    denominator = math.sqrt(len(question_tokens) * len(memory_tokens))
    return len(question_tokens & memory_tokens) / denominator if denominator else 0.0


def retrieve_memories(question: str, memory_bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(memory_bank, key=lambda memory: word_score(question, memory), reverse=True)[:TOP_K]


def build(input_path: str, output_path: str, manager_path: str, device: str, max_new_tokens: int) -> None:
    dialogues = load_dialogues(read_json(input_path))
    manager = ManagerRunner(manager_path, device)
    rows = []

    for dialogue in dialogues:
        dialogue_id = str(dialogue["dialogue_id"])
        memory_bank: list[dict[str, Any]] = []

        for turn_index, current_turn in enumerate(dialogue.get("turns", [])):
            output = manager.generate(manager_prompt(current_turn, memory_bank), max_new_tokens)
            apply_manager_output(memory_bank, output, dialogue_id, turn_index)

        for question_index, question in enumerate(dialogue.get("questions", [])):
            candidates = retrieve_memories(question["question"], memory_bank)
            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "question_id": str(question.get("question_id", question_index)),
                    "question": question["question"],
                    "retrieved_memories": candidates,
                    "answer": question["answer"],
                    "metadata": {
                        "manager_model": manager_path,
                        "top_k": TOP_K,
                        "retrieved_count": len(candidates),
                    },
                }
            )

    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} Answer tuples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Answer Agent training data with a trained Memory Manager.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    build(args.input, args.output, args.manager_model, args.device, args.max_new_tokens)


if __name__ == "__main__":
    main()
