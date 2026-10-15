"""Answer data must use the same rejected-action semantics as validation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.manager_validation import evaluate_roll_forward
from scripts import build_answer_training_data as answer_data


class AnswerDataTransitionTests(unittest.TestCase):
    def test_rejections_preserve_bank_trace_and_match_validation(self) -> None:
        outputs = [
            '{"memory":[{"event":"ADD","text":"original"}]}',
            '{"memory":[{"event":"UPDATE","id":"conv-x:m0","text":"bad"},'
            '{"event":"DELETE","id":"conv-x:m0"}]}',
            '{"memory":[',
            '{"memory":[{"event":"ADD","text":"new fact"}]}',
        ]
        timeline = {
            "dialogue_id": "conv-x",
            "participants": ["A"],
            "turns": [
                {"turn_index": index, "speaker": "A", "timestamp": f"t{index}",
                 "facts": [{"text": f"fact {index}"}]}
                for index in range(len(outputs))
            ],
            "questions": [{"question_id": "0", "question": "What?",
                           "answer": "original", "category": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            timeline_path = Path(directory) / "timeline.json"
            output_path = Path(directory) / "answer.jsonl"
            timeline_path.write_text(json.dumps([timeline]), encoding="utf-8")
            args = argparse.Namespace(
                timeline=str(timeline_path), output=str(output_path),
                manager_model="base", manager_adapter="adapter", device="cpu",
                max_new_tokens=2048, max_prompt_tokens=4096,
                manager_top_k=5, answer_top_k_per_speaker=30,
            )
            with (
                patch.object(answer_data, "Manager") as manager,
                patch("evaluation.manager_validation.top_k",
                      side_effect=lambda query, bank, k: list(bank)),
                patch.object(answer_data, "top_k_per_speaker",
                             side_effect=lambda query, bank, participants, k: list(bank)),
                self.assertLogs(answer_data.LOGGER, level="WARNING") as warnings,
            ):
                manager.return_value.generate_prompt.side_effect = outputs
                answer_data.build(args)
            self.assertEqual(len(warnings.output), 2)
            self.assertIn("duplicate target id", warnings.output[0])
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["text"] for item in row["retrieved_memories"]],
                ["original", "new fact"],
            )
            self.assertEqual(row["metadata"]["invalid_manager_turns"], 2)
            trace = [json.loads(line) for line in
                     output_path.with_suffix(".manager_trace.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
            self.assertEqual([turn["valid"] for turn in trace], [True, False, False, True])
            self.assertEqual([turn["memory_size"] for turn in trace], [1, 1, 1, 2])
            self.assertEqual([turn["raw"] for turn in trace], outputs)

            validation_banks = []

            def answer(question, bank, participants):
                validation_banks.append(bank)
                return "original"

            generated = iter(outputs)
            with patch("evaluation.manager_validation.top_k",
                       side_effect=lambda query, bank, k: list(bank)):
                result = evaluate_roll_forward(
                    [timeline], lambda prompt: next(generated), answer, 5,
                )
            self.assertEqual(result["summary"]["invalid_turns"], 2)
            self.assertEqual(validation_banks[0], row["retrieved_memories"])


if __name__ == "__main__":
    unittest.main()
