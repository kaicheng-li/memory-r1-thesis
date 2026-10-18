"""Manager validation aligned with single-step training and deployment rollout."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from evaluation.metrics import bleu1_score, exact_match, f1_score
from memfactory.manager_env import validate_and_apply
from memfactory.modules.memory_updater import build_manager_input
from scripts.construct_memory_bank import parse_manager_output, top_k


ManagerGenerate = Callable[[str], str]
AnswerGenerate = Callable[[str, list[dict[str, Any]], list[str]], str]


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def manager_transition(
    state: dict[str, Any],
    memory_bank: list[dict[str, Any]],
    manager_generate: ManagerGenerate,
    manager_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = [
        fact for fact in state.get("current_facts", state.get("facts", []))
        if isinstance(fact, dict) and str(fact.get("text", "")).strip()
    ]
    if not facts:
        return list(memory_bank), {"valid": True, "actions": 0, "error": None}
    query = " ".join(str(fact["text"]) for fact in facts)
    visible = top_k(query, memory_bank, manager_top_k)
    prompt = build_manager_input(visible, facts)
    raw = manager_generate(prompt)
    try:
        decisions = parse_manager_output(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return list(memory_bank), {
            "valid": False, "actions": 0, "error": f"parse: {exc}", "raw": raw,
        }
    result = validate_and_apply(
        memory_bank,
        visible,
        decisions,
        dialogue_id=str(state["dialogue_id"]),
        source_turn=int(state.get("target_turn", state.get("turn_index", 0))),
        speaker=str(state.get("target_speaker", state.get("speaker", ""))),
        timestamp=str(state.get("target_timestamp", state.get("timestamp", ""))),
    )
    return result.memory_bank, {
        "valid": result.valid,
        "actions": result.applied_actions,
        "error": result.error,
        "raw": raw,
    }


def _score_answers(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "qa": count,
        "em": sum(record["em"] for record in records) / count if count else 0.0,
        "f1": sum(record["f1"] for record in records) / count if count else 0.0,
        "bleu1": sum(record["bleu1"] for record in records) / count if count else 0.0,
    }


def evaluate_single_step(
    states: list[dict[str, Any]],
    manager_generate: ManagerGenerate,
    answer_generate: AnswerGenerate,
    manager_top_k: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Teacher-forced validation on Stage-0 oracle-history states."""
    state_cache: dict[tuple[str, int], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    records = []
    for index, state in enumerate(states, 1):
        label = (
            f"stage=single_step qa={index}/{len(states)} "
            f"dialogue={state['dialogue_id']} qid={state['question_id']} "
            f"target_turn={state['target_turn']}"
        )
        key = (str(state["dialogue_id"]), int(state["target_turn"]))
        if key not in state_cache:
            started = perf_counter()
            if progress:
                progress(f"{label} component=manager event=start")
            state_cache[key] = manager_transition(
                state, list(state["old_bank"]), manager_generate, manager_top_k
            )
            if progress:
                bank, action = state_cache[key]
                progress(
                    f"{label} component=manager event=done "
                    f"valid={action['valid']} actions={action['actions']} "
                    f"memory={len(bank)} elapsed_s={perf_counter()-started:.1f} "
                    f"error={action['error']!r}"
                )
        bank, action = state_cache[key]
        started = perf_counter()
        if progress:
            progress(f"{label} component=answer event=start")
        prediction = answer_generate(state["question"], bank, state["participants"])
        records.append({
            "dialogue_id": state["dialogue_id"],
            "question_id": state["question_id"],
            "question": state["question"],
            "category": state.get("category", 0),
            "gold": state["answer"],
            "prediction": prediction,
            "em": exact_match(prediction, state["answer"]),
            "f1": f1_score(prediction, state["answer"]),
            "bleu1": bleu1_score(prediction, state["answer"]),
            "manager_valid": action["valid"],
            "manager_error": action["error"],
        })
        if progress:
            progress(
                f"{label} component=answer event=done "
                f"em={records[-1]['em']:.4f} f1={records[-1]['f1']:.4f} "
                f"elapsed_s={perf_counter()-started:.1f}"
            )
    summary = _score_answers(records)
    summary.update({
        "unique_states": len(state_cache),
        "valid_states": sum(action[1]["valid"] for action in state_cache.values()),
    })
    return {"summary": summary, "records": records}


def evaluate_roll_forward(
    timelines: list[dict[str, Any]],
    manager_generate: ManagerGenerate,
    answer_generate: AnswerGenerate,
    manager_top_k: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Deployment-aligned validation: recursively build each dialogue bank."""
    records = []
    dialogue_stats = []
    for timeline in timelines:
        bank: list[dict[str, Any]] = []
        valid_turns = 0
        invalid_turns = 0
        action_count = 0
        for index, turn in enumerate(timeline["turns"], 1):
            label = (
                f"stage=roll_forward component=manager "
                f"dialogue={timeline['dialogue_id']} "
                f"turn={index}/{len(timeline['turns'])} "
                f"source_turn={turn['turn_index']}"
            )
            started = perf_counter()
            if progress:
                progress(f"{label} event=start memory={len(bank)}")
            state = {
                **turn,
                "dialogue_id": timeline["dialogue_id"],
                "current_facts": turn.get("facts", []),
                "target_turn": turn["turn_index"],
                "target_speaker": turn.get("speaker", ""),
                "target_timestamp": turn.get("timestamp", ""),
            }
            bank, action = manager_transition(
                state, bank, manager_generate, manager_top_k
            )
            if action["valid"]:
                valid_turns += 1
                action_count += action["actions"]
            else:
                invalid_turns += 1
            if progress:
                progress(
                    f"{label} event=done valid={action['valid']} "
                    f"actions={action['actions']} memory={len(bank)} "
                    f"invalid_turns={invalid_turns} "
                    f"elapsed_s={perf_counter()-started:.1f} error={action['error']!r}"
                )
        for index, question in enumerate(timeline["questions"], 1):
            label = (
                f"stage=roll_forward component=answer "
                f"dialogue={timeline['dialogue_id']} "
                f"qa={index}/{len(timeline['questions'])} qid={question['question_id']}"
            )
            started = perf_counter()
            if progress:
                progress(f"{label} event=start")
            prediction = answer_generate(
                question["question"], bank, timeline["participants"]
            )
            records.append({
                "dialogue_id": timeline["dialogue_id"],
                "question_id": question["question_id"],
                "question": question["question"],
                "category": question["category"],
                "gold": question["answer"],
                "prediction": prediction,
                "em": exact_match(prediction, question["answer"]),
                "f1": f1_score(prediction, question["answer"]),
                "bleu1": bleu1_score(prediction, question["answer"]),
            })
            if progress:
                progress(
                    f"{label} event=done em={records[-1]['em']:.4f} "
                    f"f1={records[-1]['f1']:.4f} "
                    f"elapsed_s={perf_counter()-started:.1f}"
                )
        dialogue_stats.append({
            "dialogue_id": timeline["dialogue_id"],
            "turns": len(timeline["turns"]),
            "valid_turns": valid_turns,
            "invalid_turns": invalid_turns,
            "actions": action_count,
            "final_bank_size": len(bank),
        })
    summary = _score_answers(records)
    summary.update({
        "dialogues": len(timelines),
        "valid_turns": sum(item["valid_turns"] for item in dialogue_stats),
        "invalid_turns": sum(item["invalid_turns"] for item in dialogue_stats),
        "actions": sum(item["actions"] for item in dialogue_stats),
    })
    return {"summary": summary, "dialogues": dialogue_stats, "records": records}
