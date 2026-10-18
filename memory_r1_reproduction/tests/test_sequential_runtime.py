from __future__ import annotations

import os
import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.locomo_split import select_locomo_split
from memfactory.memory_runtime import apply_decisions, parse_manager_output
from memfactory.modules.memory_retriever import (
    build_answer_input,
    build_answer_input_from_memories,
)
from scripts.build_answer_training_data import build_rows, raw_questions
from scripts.train_manager_grpo import apply_policy_update, train_from_tuples, potential_shaping_rewards, evidence_potential
from memfactory.modules.memory_updater import build_manager_input


def setUpModule():
    global _retrieval_patch
    _retrieval_patch = patch.dict(os.environ, {"MEMORY_RETRIEVAL_BACKEND": "lexical"})
    _retrieval_patch.start()


def tearDownModule():
    _retrieval_patch.stop()


class ManagerRuntimeTests(unittest.TestCase):
    def test_parser_skips_invalid_prefix_and_handles_braces_in_string(self):
        value = 'reasoning {bad json} then {"memory":[{"event":"ADD","text":"literal { brace"}]}'
        self.assertEqual(
            parse_manager_output(value),
            [{"event": "ADD", "text": "literal { brace"}],
        )

    def test_invalid_later_action_does_not_partially_mutate_bank(self):
        bank = [{
            "id": "conv-26:m0",
            "text": "old memory",
            "speaker": "A",
            "timestamp": "t0",
            "source_turn_ids": ["D1:1"],
        }]
        before = [dict(item) for item in bank]
        with self.assertRaises(ValueError):
            apply_decisions(
                bank,
                [
                    {"event": "UPDATE", "id": "conv-26:m0", "text": "updated memory"},
                    {"event": "DELETE", "id": "hidden-id"},
                ],
                "conv-26",
                {"dia_id": "D1:2", "speaker": "A", "timestamp": "t1"},
                visible_memory_ids={"conv-26:m0"},
            )
        self.assertEqual(bank, before)

    def test_cannot_update_memory_not_visible_in_prompt(self):
        bank = [{"id": "conv-26:m0", "text": "hidden", "source_turn_ids": []}]
        with self.assertRaisesRegex(ValueError, "not visible"):
            apply_decisions(
                bank,
                [{"event": "UPDATE", "id": "conv-26:m0", "text": "new"}],
                "conv-26",
                {"dia_id": "D1:2"},
                visible_memory_ids=set(),
            )

    def test_dialogue_split_freezes_raw_question_id_before_category_filter(self):
        sample = {
            "sample_id": "conv-26",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "fact"}],
            },
            "qa": [
                {"category": 5, "question": "adversarial", "answer": "x", "evidence": ["D1:1"]},
                {"category": 0, "question": "unclassified", "answer": "x", "evidence": ["D1:1"]},
                {"category": 1, "question": "valid", "answer": "y", "evidence": ["D1:1"]},
            ],
        }
        selected = select_locomo_split([sample], "train")
        self.assertEqual([question["question_id"] for question in selected[0]["qa"]], ["2"])

    def test_answer_tuple_keeps_list_answer_and_speaker_prompt(self):
        question = raw_questions({
            "qa": [{"question_id": "7", "question": "What?", "answer": ["A", "B"]}]
        })[0]
        self.assertEqual(question["answer"], "A, B")
        memories = [
            {"speaker": "Ada", "text": "A"},
            {"speaker": "Bob", "text": "B"},
        ]
        self.assertEqual(
            build_answer_input_from_memories("What?", memories, ["Ada", "Bob"]),
            build_answer_input("What?", {"Ada": memories[:1], "Bob": memories[1:]}),
        )
        privileged = build_answer_input_from_memories(
            "What?", [{"speaker": "Ada", "text": "A", "source_turn_ids": ["D1:7"]}], ["Ada"]
        )
        self.assertNotIn("source_turn_ids", privileged)
        self.assertNotIn("D1:7", privileged)
        manager_prompt = build_manager_input(
            [{"id": "m0", "text": "fact", "source_turn_ids": ["D1:7"]}], []
        )
        self.assertNotIn("source_turn_ids", manager_prompt)
        self.assertNotIn("D1:7", manager_prompt)

    def test_answer_rows_use_full_dialogue_bank(self):
        class Manager:
            def generate(self, _bank, facts, _top_k):
                text = facts[0]["text"]
                return json.dumps({"memory": [{"event": "ADD", "text": text}]})

        tuple_row = {
            "dialogue_id": "conv-26",
            "split": "train",
            "turn_index": 1,
            "participants": ["Ada", "Bob"],
            "dialogue_turns": [{
                "turn_index": 1,
                "dia_id": "D1:2",
                "speaker": "Ada",
                "timestamp": "t2",
                "facts": [{"speaker": "Ada", "text": "Ada met Bob"}],
            }],
            "linked_questions": [{
                "question_id": "9",
                "question": "Who did Ada meet?",
                "answer": "Bob",
                "evidence": ["D1:2"],
            }],
        }
        timeline = dict(tuple_row, turns=tuple_row["dialogue_turns"], questions=tuple_row["linked_questions"])
        # A later fact after the question's evidence target must still be in
        # the full-dialogue Answer bank (baseline downstream protocol).
        timeline["turns"] = timeline["turns"] + [{"turn_index": 2, "dia_id": "D1:3", "speaker": "Ada",
            "timestamp": "t3", "facts": [{"text": "Ada later moved to Rome"}]}]
        rows = build_rows([timeline], Manager(), 5, 30, "train", "base-model")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question_id"], "9")
        self.assertEqual(rows[0]["participants"], ["Ada", "Bob"])
        self.assertEqual(rows[0]["retrieved_memories"][0]["source_turn_ids"], ["D1:2"])
        self.assertEqual(rows[0]["metadata"]["turns"], 2)
        self.assertEqual(len(rows[0]["retrieved_memories"]), 2)
        self.assertEqual(rows[0]["metadata"]["source"], "full_dialogue_cached_facts_greedy")

    def test_invalid_manager_action_keeps_prior_bank_and_reward_alignment(self):
        class Manager:
            def __init__(self):
                self.count = 0

            def generate(self, *_args):
                self.count += 1
                if self.count % 2:
                    return '{"memory":[{"event":"ADD","text":"fact one"}]}', torch.tensor([1])
                return "malformed", torch.tensor([2])

            def log_probability(self, *_args):
                return torch.tensor([0.0])

        class Answer:
            def generate(self, *_args, **_kwargs):
                return "Answer: yes", torch.tensor([1])

        turns = [
            {"dia_id": "D1:1", "speaker": "Ada", "timestamp": "t1", "facts": [{"text": "fact one"}]},
            {"dia_id": "D1:2", "speaker": "Ada", "timestamp": "t2", "facts": [{"text": "fact two"}]},
        ]
        row = {
            "dialogue_id": "conv-26",
            "participants": ["Ada"],
            "dialogue_turns": turns,
            "linked_questions": [{"question": "fact one", "answer": "yes", "evidence": ["D1:1"]}],
        }
        args = SimpleNamespace(
            algorithm="grpo", num_generations=2, manager_top_k=5,
            max_new_tokens=16, temperature=1.0, invalid_action_penalty=0.1,
            answer_top_k=30, answer_max_new_tokens=16, answer_temperature=0.0,
            evidence_weight=0.5, gamma=1.0,
        )
        captured = []

        def capture(_manager, _reference, _optimizer, trajectories, rewards, _args):
            captured.append((trajectories, rewards))
            return float("nan")

        with patch("scripts.train_manager_grpo.apply_policy_update", side_effect=capture):
            train_from_tuples(args, Manager(), None, Answer(), None, lambda *_: 1.0, [row])
        trajectories, rewards = captured[0]
        self.assertEqual(len(trajectories), 2)
        for trajectory, reward in zip(trajectories, rewards):
            self.assertEqual(len(trajectory["actions"]), len(trajectory["memory_states"]))
            self.assertEqual([action["valid"] for action in trajectory["actions"]], [True, False])
            self.assertEqual(trajectory["memory_states"][1], trajectory["memory_states"][0])
            self.assertAlmostEqual(sum(reward), 1.5)
            self.assertAlmostEqual(reward[1], 1.0)
            from memfactory.sequential_credit import question_credit
            _, groups = question_credit(trajectories, [rewards])
            self.assertEqual(groups[1]["returns"], [0.0, 0.0])
            self.assertEqual(groups[0]["returns"], [1.5, 1.5])


class ZeroVarianceTests(unittest.TestCase):
    def test_zero_variance_group_skips_policy_and_kl_update(self):
        class Manager:
            device = "cpu"
            model = torch.nn.Linear(1, 1)

            def log_probability(self, *_args):
                raise AssertionError("zero-variance group must not be scored")

        class Optimizer:
            stepped = False

            def zero_grad(self):
                raise AssertionError("zero-variance group must not zero gradients")

            def step(self):
                self.stepped = True

        manager = Manager()
        optimizer = Optimizer()
        trajectories = [
            {"actions": [{"turn_position": 0}]},
            {"actions": [{"turn_position": 0}]},
        ]
        args = SimpleNamespace(gamma=1.0, variance_epsilon=1e-6)
        result = apply_policy_update(manager, object(), optimizer, trajectories, [[0.0], [0.0]], args)
        self.assertTrue(math.isnan(result))
        self.assertFalse(optimizer.stepped)

    def test_nonconstant_group_backpropagates_each_action_before_the_next(self):
        class Manager:
            device = "cpu"

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()
                self.saw_prior_gradient = False

            def log_probability(self, prompt, _completion_ids):
                if prompt == "second":
                    self.saw_prior_gradient = self.model.weight.grad is not None
                factor = 1.0 if prompt == "first" else 2.0
                return self.model(torch.tensor([[factor]])).flatten()

        manager = Manager()
        optimizer = torch.optim.SGD(manager.model.parameters(), lr=0.1)
        trajectories = [
            {"actions": [{"turn_position": 0, "prompt": name,
                          "completion_ids": torch.tensor([1]),
                          "old_token_log_probs": torch.tensor([0.0])}]}
            for name in ("first", "second")
        ]
        args = SimpleNamespace(
            gamma=1.0, variance_epsilon=1e-6, algorithm="grpo", clip_epsilon=0.2,
        )
        apply_policy_update(manager, None, optimizer, trajectories, [[0.0], [1.0]], args)
        self.assertTrue(manager.saw_prior_gradient)
        self.assertGreater(manager.model.weight.item(), 0.0)


class CompletionTruncationTests(unittest.TestCase):
    def test_answer_completions_stop_at_first_eos(self):
        from scripts.train_answer_grpo import generate_completions

        class FakeBatch(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, *_args, **_kwargs):
                return "formatted prompt"

            def __call__(self, *_args, **_kwargs):
                return FakeBatch(input_ids=torch.tensor([[5]]))

            def decode(self, ids, **_kwargs):
                return ",".join(str(int(item)) for item in ids)

        class FakeModel:
            training = True

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def generate(self, **_kwargs):
                assert _kwargs["top_k"] == 0 and _kwargs["top_p"] == 1.0
                return torch.tensor([[5, 7, 2, 0], [5, 8, 9, 2]])

        completions = generate_completions(
            FakeModel(), FakeTokenizer(), "prompt", 2, 4, 1.0, "cpu", 32
        )
        self.assertEqual([ids.tolist() for _, ids in completions], [[7, 2], [8, 9, 2]])

    def test_left_padding_preserves_generation_positions_and_action_mask(self):
        from scripts.train_answer_grpo import score_group

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, *_args, **_kwargs):
                return "prompt"

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": torch.tensor([[4, 5]])}

        class Decoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask, position_ids, **_kwargs):
                self.positions = position_ids
                return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 4)))

        class PositionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Decoder()
                self.lm_head = torch.nn.Linear(4, 10)

        model = PositionModel()
        _, mask = score_group(
            model, Tokenizer(), "prompt", [torch.tensor([7, 2]), torch.tensor([8, 9, 2])], "cpu", 2, 32
        )
        self.assertEqual(model.model.positions.tolist(), [[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 3])



class ShapingBoundaryTests(unittest.TestCase):
    def test_provenance_proxy_can_reward_semantically_wrong_text(self):
        # This documents a known scientific limitation, not semantic evidence validation.
        potential = evidence_potential(
            {"question": "Where does Ada live?", "evidence": ["D1:1"]},
            [{"speaker": "Ada", "text": "unrelated garbage", "source_turn_ids": ["D1:1"]}],
            30, ["Ada"], {"D1:1"},
        )
        self.assertEqual(potential, 1.0)

    def test_nonzero_terminal_potential_can_reverse_qa_preference(self):
        qa_a, qa_b, eta = 0.8, 0.6, 0.5
        total_a = qa_a + eta * sum(potential_shaping_rewards([0.0], 1.0))
        total_b = qa_b + eta * sum(potential_shaping_rewards([1.0], 1.0))
        self.assertGreater(qa_a, qa_b)
        self.assertLess(total_a, total_b)

    def test_zero_terminal_shaping_cancels_in_same_state_group_advantage(self):
        # Two different futures from the same state (Phi=0.3).
        shaped = [
            sum(potential_shaping_rewards(path, 1.0, "zero", 0.3))
            for path in ([0.8, 1.0], [0.1, 0.0])
        ]
        self.assertAlmostEqual(shaped[0], -0.3)
        self.assertAlmostEqual(shaped[1], -0.3)
        base = torch.tensor([0.2, 0.7], dtype=torch.float64)
        with_shaping = base + torch.tensor(shaped, dtype=torch.float64)
        torch.testing.assert_close(base - base.mean(), with_shaping - with_shaping.mean())

    def test_discounted_telescoping_identity(self):
        gamma, initial, path = 0.9, 0.2, [0.4, 0.8, 0.6]
        rewards = potential_shaping_rewards(path, gamma, "observed", initial)
        discounted = sum(gamma ** index * reward for index, reward in enumerate(rewards))
        self.assertAlmostEqual(discounted, -initial + gamma ** len(path) * path[-1])


class FixedExtractorEvaluationTests(unittest.TestCase):
    def test_cached_facts_do_not_call_extractor_and_invalid_actions_continue(self):
        from scripts.construct_memory_bank import construct

        class Manager:
            count = 0

            def extract(self, _turn):
                raise AssertionError("cached facts must not invoke a policy-dependent extractor")

            def generate(self, *_args):
                self.count += 1
                return "malformed" if self.count == 1 else '{"memory":[{"event":"ADD","text":"valid fact"}]}'

        turns = [
            {"dia_id": f"D1:{index}", "speaker": "A", "facts": [{"text": "fixed fact"}]}
            for index in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "output.json"
            source.write_text(json.dumps([{"dialogue_id": "conv-26", "turns": turns}]), encoding="utf-8")
            with patch("scripts.construct_memory_bank.Manager", return_value=Manager()):
                construct(str(source), str(output), "fake", "cpu", 5, 16, None)
            record = json.loads(output.read_text(encoding="utf-8"))[0]
            self.assertEqual(record["fact_source"], "cached")
            self.assertEqual(record["invalid_actions"], 1)
            self.assertEqual(len(record["memory_bank"]), 1)
            self.assertEqual(record["memory_bank"][0]["source_turn_ids"], ["D1:2"])


class Stage0SequentialAdapterTests(unittest.TestCase):
    def test_full_prefix_recovers_long_evidence_without_future_turns(self):
        from scripts.build_sequential_from_stage0 import build_rows as adapt

        turns = [{"dia_id": f"D1:{i}", "turn_index": i, "facts": []} for i in range(5)]
        question = {"question_id": "3", "question": "what?", "answer": "yes", "category": 1}
        timeline = {"dialogue_id": "conv-26", "split": "train", "participants": ["A"],
                    "turns": turns, "questions": [question]}
        state = dict(question, dialogue_id="conv-26", split="train", target_turn=3,
                     evidence=["D1:0", "D1:3"])
        short, short_manifest = adapt([timeline], [state], "train", 1)
        full, full_manifest = adapt([timeline], [state], "train", -1)
        self.assertEqual(short, [])
        self.assertEqual(len(short_manifest["omitted_out_of_window"]), 1)
        self.assertEqual([turn["turn_index"] for turn in full[0]["dialogue_turns"]], [0, 1, 2, 3])
        self.assertEqual(full_manifest["linked_questions"], 1)
        self.assertEqual(full[0]["temporal_memory_bank"], [])
        with self.assertRaisesRegex(ValueError, "QA identity/content mismatch"):
            adapt([timeline], [dict(state, answer="wrong")], "train", -1)


class SequentialValidationTests(unittest.TestCase):
    def test_validation_uses_full_bank_and_continues_after_invalid_action(self):
        from memfactory.sequential_validation import evaluate_roll_forward
        from scripts.train_manager_grpo import normalize_text, exact_match_reward, answer_reward

        timeline = {"dialogue_id": "conv-30", "participants": ["A"], "turns": [
            {"dia_id": f"D1:{i}", "speaker": "A", "facts": [{"text": f"fact {i}"}]}
            for i in (1, 2)
        ], "questions": [{"question_id": "0", "question": "what?", "answer": "yes yes", "category": 1}]}
        outputs = iter(["malformed", '{"memory":[{"event":"ADD","text":"last turn"}]}'])
        events = []

        def manager(_prompt):
            self.assertEqual(events[-1][0], "validation_turn_start")
            return next(outputs)

        def answer(_question, memories, _participants):
            self.assertEqual(events[-1][0], "validation_answer_start")
            self.assertEqual(memories[0]["text"], "last turn")
            return "yes yes extra"

        result = evaluate_roll_forward(
            [timeline], manager, answer,
            SimpleNamespace(manager_top_k=5, answer_top_k=30),
            normalize_text, exact_match_reward, answer_reward,
            progress=lambda event, **fields: events.append((event, fields)),
        )
        self.assertAlmostEqual(result["summary"]["token_f1_multiset"], 0.8)
        self.assertAlmostEqual(result["summary"]["reward_f1_unique_overlap"], 0.4)
        self.assertEqual(result["dialogues"][0]["invalid_actions"], 1)
        self.assertEqual([event for event, _ in events], [
            "validation_turn_start", "validation_turn_done",
            "validation_turn_start", "validation_turn_done",
            "validation_answer_start", "validation_answer_done",
        ])
        self.assertEqual(events[1][1]["invalid_actions"], 1)


if __name__ == "__main__":
    unittest.main()
