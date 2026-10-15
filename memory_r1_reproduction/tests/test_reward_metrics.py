"""Deterministic tests for the shared LoCoMo reward and diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.matching import match_locomo_v2
from evaluation.manager_validation import evaluate_single_step
from evaluation.metrics import f1_score
from memfactory.locomo_split import questions_with_raw_ids, select_locomo_split
from memfactory.manager_env import validate_and_apply


class TokenF1Tests(unittest.TestCase):
    def test_exact_match_is_one(self) -> None:
        self.assertEqual(f1_score("blue bicycle", "blue bicycle"), 1.0)

    def test_partial_overlap_is_fractional(self) -> None:
        self.assertAlmostEqual(f1_score("blue bicycle", "blue car"), 0.5)

    def test_articles_case_and_punctuation_are_normalized(self) -> None:
        self.assertEqual(f1_score("The Blue, bicycle!", "blue bicycle"), 1.0)

    def test_no_overlap_is_zero(self) -> None:
        self.assertEqual(f1_score("red car", "blue bicycle"), 0.0)


class LayeredDiagnosticTests(unittest.TestCase):
    def test_equivalent_date_formats_match(self) -> None:
        result = match_locomo_v2("7 May 2023", "May 7th, 2023")
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["tier"], "date")

    def test_conflicting_date_cannot_pass_f1_tier(self) -> None:
        result = match_locomo_v2("8 May 2023", "7 May 2023")
        self.assertGreaterEqual(result["f1"], 0.6)
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["tier"], "none")


class DataIdentityTests(unittest.TestCase):
    def test_raw_question_id_survives_category_filtering(self) -> None:
        sample = {
            "sample_id": "conv-30",
            "qa": [
                {"question": "first", "category": 1},
                {"question": "excluded", "category": 5},
                {"question": "third", "category": 4},
            ],
        }
        self.assertEqual(
            [row["question_id"] for row in questions_with_raw_ids(sample)],
            ["0", "1", "2"],
        )
        selected = select_locomo_split([sample], "validation")
        self.assertEqual(
            [row["question_id"] for row in selected[0]["qa"]], ["0", "2"]
        )


class ManagerEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = [{
            "id": "conv-1:m0", "text": "old", "speaker": "A",
            "timestamp": "t0", "source_turn": 0,
        }]

    def test_illegal_hidden_id_is_rejected_atomically(self) -> None:
        result = validate_and_apply(
            self.bank, [], [{"event": "DELETE", "id": "conv-1:m0"}],
            dialogue_id="conv-1", source_turn=1, speaker="A", timestamp="t1",
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.memory_bank, self.bank)

    def test_valid_add_gets_provenance(self) -> None:
        result = validate_and_apply(
            self.bank, self.bank, [{"event": "ADD", "text": "new"}],
            dialogue_id="conv-1", source_turn=1, speaker="B", timestamp="t1",
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.memory_bank[-1]["id"], "conv-1:m5")
        self.assertEqual(result.memory_bank[-1]["speaker"], "B")
        self.assertEqual(result.memory_bank[-1]["source_turn"], 1)

    def test_duplicate_target_is_rejected_atomically(self) -> None:
        decisions = [
            {"event": "UPDATE", "id": "conv-1:m0", "text": "changed"},
            {"event": "DELETE", "id": "conv-1:m0"},
        ]
        result = validate_and_apply(
            self.bank, self.bank, decisions,
            dialogue_id="conv-1", source_turn=1, speaker="A", timestamp="t1",
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.memory_bank, self.bank)


class ManagerValidationTests(unittest.TestCase):
    def test_single_step_uses_candidate_bank(self) -> None:
        state = {
            "dialogue_id": "conv-x",
            "question_id": "0",
            "question": "When?",
            "answer": "7 May 2023",
            "participants": ["A", "B"],
            "target_turn": 0,
            "target_speaker": "A",
            "target_timestamp": "8 May 2023",
            "current_facts": [{"text": "The event was on 7 May 2023"}],
            "old_bank": [],
        }

        def manager_generate(_prompt: str) -> str:
            return '{"memory":[{"event":"ADD","text":"The event was on 7 May 2023"}]}'

        def answer_generate(_question, bank, _participants) -> str:
            return "7 May 2023" if bank else "unknown"

        result = evaluate_single_step(
            [state], manager_generate, answer_generate, manager_top_k=5
        )
        self.assertEqual(result["summary"]["unique_states"], 1)
        self.assertEqual(result["summary"]["valid_states"], 1)
        self.assertEqual(result["summary"]["f1"], 1.0)

if __name__ == "__main__":
    unittest.main()
