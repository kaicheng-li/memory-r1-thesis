"""Full-dialogue validation using fixed cached facts and the shared runtime."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from memfactory.locomo_split import SPLIT_DIALOGUE_IDS
from memfactory.memory_runtime import apply_decisions, parse_manager_output, retrieve, retrieve_per_speaker
from memfactory.modules.memory_updater import build_manager_input


def multiset_f1(prediction: str, gold: str, normalize) -> float:
    predicted, expected = normalize(prediction).split(), normalize(gold).split()
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2.0 * overlap / (len(predicted) + len(expected))


def load_validation_timelines(path: str) -> list[dict]:
    timelines = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [str(item["dialogue_id"]) for item in timelines]
    if sorted(ids) != sorted(SPLIT_DIALOGUE_IDS["validation"]):
        raise ValueError("Checkpoint selection requires exactly the validation dialogues.")
    for timeline in timelines:
        if timeline.get("split") != "validation":
            raise ValueError("Validation timeline has an unexpected split label.")
        if any(not isinstance(turn.get("facts"), list) for turn in timeline["turns"]):
            raise ValueError("Validation requires cached fixed-extractor facts on every turn.")
    return timelines


def evaluate_roll_forward(timelines, manager_generate, answer_generate, args, normalize, em, reward_f1):
    """Build each full bank once, then answer every QA greedily.

    No evidence annotations are consumed. This follows full-dialogue QA,
    rather than the evidence-targeted training-prefix diagnostic.
    """
    records, dialogues = [], []
    for timeline in timelines:
        bank, invalid, attempted = [], 0, 0
        for turn in timeline["turns"]:
            facts = [fact for fact in turn["facts"]
                     if isinstance(fact, dict) and str(fact.get("text", "")).strip()]
            if not facts:
                continue
            visible = retrieve(" ".join(str(fact["text"]) for fact in facts), bank, args.manager_top_k)
            raw = manager_generate(build_manager_input(visible, facts))
            attempted += 1
            try:
                apply_decisions(bank, parse_manager_output(raw), timeline["dialogue_id"], turn,
                                visible_memory_ids={item["id"] for item in visible})
            except (ValueError, KeyError, TypeError):
                invalid += 1
        for question in timeline["questions"]:
            if int(question.get("category", 0)) not in {1, 2, 3, 4}:
                continue
            gold = question["answer"]
            if isinstance(gold, list):
                gold = ", ".join(str(value) for value in gold)
            memories = retrieve_per_speaker(question["question"], bank, timeline["participants"], args.answer_top_k)
            prediction = answer_generate(question["question"], memories, timeline["participants"])
            records.append({
                "dialogue_id": timeline["dialogue_id"], "question_id": str(question["question_id"]),
                "category": question["category"], "gold": gold, "prediction": prediction,
                "em": em(prediction, gold), "token_f1_multiset": multiset_f1(prediction, gold, normalize),
                "reward_f1_unique_overlap": reward_f1(prediction, gold),
            })
        dialogues.append({"dialogue_id": timeline["dialogue_id"], "invalid_actions": invalid,
                          "attempted_actions": attempted, "memory_size": len(bank)})
    if not records:
        raise ValueError("Validation produced no eligible QA.")
    summary = {name: sum(row[name] for row in records) / len(records)
               for name in ("em", "token_f1_multiset", "reward_f1_unique_overlap")}
    summary.update({"qa": len(records), "protocol": "full_dialogue_cached_facts_greedy"})
    return {"summary": summary, "dialogues": dialogues, "records": records}
