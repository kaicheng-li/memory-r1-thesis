"""Bounded recovery of Manager JSON without inventing action content."""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memfactory.memory_runtime import apply_decisions, parse_manager_output


class ManagerParserTests(unittest.TestCase):
    def setUp(self):
        self.actions = [{"event": "ADD", "text": "Paris", "tags": ["city"]}]
        self.complete = json.dumps({"memory": self.actions})
        self.missing_brace = self.complete[:-1]

    def test_valid_objects_and_wrappers_keep_existing_results(self):
        for text in (
            self.complete,
            "reasoning {bad json} then " + self.complete,
            "```json\n" + self.complete + "\n```",
            self.complete + '\n{"memory":[]}',
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_manager_output(text), self.actions)
        self.assertEqual(parse_manager_output('{"memory":[]}'), [])

    def test_recovers_only_outer_brace_with_common_noisy_endings(self):
        for tail in ("", " \n", "\n```", "anolizado", ".executeUpdate()}"):
            with self.subTest(tail=tail):
                self.assertEqual(
                    parse_manager_output(self.missing_brace + tail), self.actions
                )
        self.assertEqual(parse_manager_output('{"memory":[]'), [])
        self.assertEqual(
            parse_manager_output('```json\n{ \n "memory" : ' + json.dumps(self.actions)),
            self.actions,
        )

    def test_nested_arrays_and_json_punctuation_inside_strings_are_preserved(self):
        actions = [{
            "event": "ADD", "text": 'literal ] } { "quote" \\ slash',
            "tags": ["[x]", {"nested": ["}", "]"]}],
        }]
        self.assertEqual(
            parse_manager_output(json.dumps({"memory": actions})[:-1]), actions
        )

    def test_recovers_original_actions_before_echoed_empty_example(self):
        echoed = '\nuser\nIf nothing changes, return {"memory":[]}.\n'
        self.assertEqual(
            parse_manager_output(self.missing_brace + echoed), self.actions
        )

    def test_incomplete_action_content_is_not_completed_or_replaced_by_example(self):
        for broken in (
            '{"memory":[{"event":"ADD","text":"unfinished',
            '{"memory":[{"event":"ADD","text":"Paris"',
            '{"memory":[{"event":"ADD","text":"Paris"}',
            '{"memory":[{"event":"ADD","text":"Paris",}]]',
            '{"memory":[{"event":"ADD","text":"Paris"},]',
        ):
            for tail in ("", '\nuser\nIf nothing changes, return {"memory":[]}.'):
                with self.subTest(broken=broken, tail=tail):
                    with self.assertRaisesRegex(ValueError, "valid memory JSON"):
                        parse_manager_output(broken + tail)

    def test_broken_root_fields_and_wrong_schema_are_not_repaired(self):
        for text in (
            self.missing_brace + ', "other":',
            self.missing_brace + ' "other": 1}',
            self.missing_brace + "]",
            self.missing_brace + ":",
            '{"memory":{}',
            '{"memory":null',
            '{"other":[]}',
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "valid memory JSON"):
                    parse_manager_output(text)

    def test_recovered_actions_still_pass_original_atomic_validation(self):
        bank = [{"id": "d:m0", "text": "Rome", "speaker": "A"}]
        before = copy.deepcopy(bank)
        turn = {"turn_index": 1, "speaker": "A", "timestamp": "now"}
        for actions in (
            [{"event": "ADD", "text": ""}],
            [{"event": "UNSUPPORTED", "text": "Paris"}],
            [{"event": "UPDATE", "id": "d:m0", "text": "Paris"},
             {"event": "DELETE", "id": "invisible"}],
        ):
            with self.subTest(actions=actions):
                decisions = parse_manager_output(json.dumps({"memory": actions})[:-1])
                with self.assertRaises(ValueError):
                    apply_decisions(bank, decisions, "d", turn)
                self.assertEqual(bank, before)
        apply_decisions(bank, parse_manager_output(self.missing_brace), "d", turn)
        self.assertEqual(bank[-1]["text"], "Paris")


if __name__ == "__main__":
    unittest.main()
