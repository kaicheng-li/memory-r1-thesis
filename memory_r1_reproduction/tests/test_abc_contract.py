"""CPU contract checks against the pinned baseline trainer, not a restatement."""
import copy
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_manager_grpo as sequential
from memfactory.sequential_credit import question_credit, clipped_surrogate
from evaluation.metrics import f1_score

BASELINE = "87f13c579c3121fe9a76119ac5a15b059681954d"


def actions(valid=(True, True)):
    return [{"actions": [{"turn_position": t, "valid": v, "completion_ids": torch.tensor([1])}
                         for t, v in enumerate(valid)]} for _ in range(2)]


class CreditContractTests(unittest.TestCase):
    def test_primary_defaults_are_locked_to_the_agreed_baseline(self):
        args = sequential.build_parser().parse_args([
            "--data-path", "unused", "--manager-model", "base", "--answer-model", "base"])
        self.assertEqual((args.reward, args.advantage_estimator, args.control_gate), ("f1", "grpo", "off"))
        self.assertEqual((args.num_generations, args.epochs, args.learning_rate), (4, 5, 1e-5))
        self.assertEqual((args.warmup_steps, args.weight_decay, args.beta), (10, .01, .01))
        self.assertEqual(args.context_overflow, "truncate")
        self.assertFalse(hasattr(args, "invalid_action_penalty"))

    def test_opposite_question_signals_are_not_erased_by_reward_averaging(self):
        trs = actions((True,))
        terms, groups = question_credit(trs, [[[1.0], [0.0]], [[0.0], [1.0]]])
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(not g["skipped"] for g in groups))
        self.assertGreater(groups[0]["advantages"][0], 0)
        self.assertLess(groups[1]["advantages"][0], 0)
        _, averaged = question_credit(trs, [[[0.5], [0.5]]])
        self.assertTrue(averaged[0]["skipped"])
        self.assertEqual(len(terms[(0, 0)]), 2)

    def test_clip_each_question_before_averaging(self):
        lp = torch.tensor([math.log(1.5)], requires_grad=True)
        old = torch.zeros(1)
        loss = (clipped_surrogate(lp, old, 1.0, .2) + clipped_surrogate(lp, old, -1.0, .2)) / 2
        self.assertAlmostEqual(float(loss), .15, places=6)
        self.assertEqual(float(clipped_surrogate(lp, old, 0.0, .2)), 0.0)

    def test_invalid_return_zero_does_not_erase_earlier_valid_credit(self):
        trs = actions((True, False))
        _, groups = question_credit(trs, [[[0.0, .8], [0.0, .2]]])
        torch.testing.assert_close(torch.tensor(groups[0]["returns"]), torch.tensor([.8, .2]))
        self.assertEqual(groups[1]["returns"], [0.0, 0.0])
        self.assertTrue(groups[1]["skipped"])

    def test_invalid_zero_is_in_normalization_not_removed_from_group(self):
        trs = actions((True,))
        trs[1]["actions"][0]["valid"] = False
        _, groups = question_credit(trs, [[[.8], [.8]]])
        self.assertLess(groups[0]["advantages"][1], 0)
        self.assertFalse(groups[0]["skipped"])

    def test_terminal_only_returns_equal_at_each_valid_turn(self):
        _, groups = question_credit(actions(), [[[0.0, 1.0], [0.0, .2]]])
        self.assertEqual(groups[0]["advantages"], groups[1]["advantages"])

    def test_zero_signal_question_keeps_its_zero_loss_weight(self):
        terms, _ = question_credit(actions((True,)), [[[1.0], [0.0]], [[.5], [.5]]])
        self.assertEqual(sum(weight for values in terms.values() for _, _, weight in values), .5)

    def test_coverage_empty_and_growing_denominator(self):
        q = {"evidence": ["d0", "d1"]}
        bank = [{"source_turn_ids": ["d0"], "text": "fact"}]
        phi = lambda seen: sequential.evidence_potential(q, bank, 0, [], seen)
        self.assertEqual(phi(set()), 0.0)
        self.assertEqual(phi({"d0"}), 1.0)
        self.assertEqual(phi({"d0", "d1"}), .5)


class PinnedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def source_at(path):
            return subprocess.check_output(["git", "show", f"{BASELINE}:memory_r1_reproduction/{path}"],
                                           cwd=ROOT, text=True, encoding="utf-8")
        source = source_at("scripts/train_manager_grpo.py")
        cls.baseline = types.ModuleType("pinned_baseline_manager")
        cls.baseline.__file__ = str(ROOT / "scripts/train_manager_grpo.py")
        sys.modules[cls.baseline.__name__] = cls.baseline
        exec(compile(source, cls.baseline.__file__, "exec"), cls.baseline.__dict__)
        # Pin the dependencies being compared too: importing current prompt /
        # transition code in both trainers would make the test self-confirming.
        env = types.ModuleType("pinned_baseline_env")
        sys.modules[env.__name__] = env
        exec(source_at("memfactory/manager_env.py"), env.__dict__)
        cls.baseline.validate_and_apply = env.validate_and_apply
        for path, name in [("memfactory/modules/memory_updater.py", "build_manager_input"),
                           ("scripts/construct_memory_bank.py", "parse_manager_output")]:
            tree = ast.parse(source_at(path))
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            namespace = {"json": json}
            exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
            setattr(cls.baseline, name, namespace[name])

    def test_one_step_transition_prompt_score_advantage_loss_and_update(self):
        self.compare_one_step(beta=0.0)

    def test_one_step_kl_reference_and_update_match_baseline(self):
        self.compare_one_step(beta=.01)

    def compare_one_step(self, beta):
        outputs = [
            '{"memory":[{"event":"ADD","text":"Paris"}]}',
            'invalid JSON',
            '{"memory":[]}',
            '{"memory":[{"event":"UPDATE","id":"conv-26:m0","text":"London"}]}',
        ]
        class Policy:
            device = "cpu"
            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                torch.nn.init.constant_(self.model.weight, .1)
                self.reference_mode = False
                self.model.disable_adapter = self.disable_adapter
                self.index, self.prompts = 0, []
            @contextmanager
            def disable_adapter(self):
                self.reference_mode = True
                try:
                    yield
                finally:
                    self.reference_mode = False
            def generate(self, prompt, *_a):
                self.prompts.append(prompt)
                value = outputs[self.index]
                self.index += 1
                return value, torch.tensor([self.index])
            def log_probability(self, _prompt, tokens, *_a):
                if self.reference_mode:
                    return torch.zeros(len(tokens))
                return self.model(tokens.float().reshape(-1, 1)).flatten()
        args = sequential.build_parser().parse_args([
            "--data-path", "unused", "--manager-model", "fake", "--answer-model", "fake",
            "--evidence-weight", "0", "--beta", str(beta)])
        old = [{"id": "conv-26:m0", "text": "Rome", "speaker": "Ada", "timestamp": "t0", "source_turn": 0}]
        facts = [{"text": "Paris"}]
        turn = {"turn_index": 8, "dia_id": "d8", "speaker": "Ada", "timestamp": "t8", "facts": facts}
        q = {"question_id": "0", "question": "Where?", "answer": "Paris", "evidence": ["d8"]}
        state = dict(q, old_bank=old, current_facts=facts, dialogue_id="conv-26", target_turn=8,
                     target_speaker="Ada", target_timestamp="t8", participants=["Ada"])
        row = {"dialogue_id": "conv-26", "participants": ["Ada"], "initial_memory_bank": old,
               "dialogue_turns": [turn], "linked_questions": [q]}
        observed_banks = []
        def reader(_a, _q, bank, *_rest):
            observed_banks.append(copy.deepcopy(bank))
            return "Paris" if any(m["text"] == "Paris" for m in bank) else "Rome"
        a, b = Policy(), Policy()
        opt_a = torch.optim.SGD(a.model.parameters(), lr=.1)
        opt_b = torch.optim.SGD(b.model.parameters(), lr=.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda _: 1.0)
        captured = {}
        real_update = sequential.apply_policy_update
        def update(*values):
            captured["trajectories"] = values[3]
            captured["loss"] = real_update(*values)
            return captured["loss"]
        with patch.dict(os.environ, {"MEMORY_RETRIEVAL_BACKEND": "lexical"}), \
             patch.object(self.baseline, "answer_from_bank", side_effect=reader), \
             patch.object(sequential, "answer_with_model", side_effect=reader), \
             patch.object(sequential, "apply_policy_update", side_effect=update):
            rolls = self.baseline.rollout_group(a, object(), state, args)
            result = self.baseline.update_policy(a, opt_a, sched, rolls, args, 1)
            sequential.train_from_tuples(args, b, None, object(), opt_b, f1_score, [row])
        self.assertEqual(a.prompts, b.prompts)
        # Invalid one-step candidates do not call the reader in either path.
        self.assertEqual(len(observed_banks), 6)
        public = lambda bank: [{k: v for k, v in m.items() if k != "source_turn_ids"} for m in bank]
        self.assertEqual([public(x) for x in observed_banks[:3]], [public(x) for x in observed_banks[3:]])
        trs = captured["trajectories"]
        self.assertEqual(public(trs[1]["memory"]), old)
        self.assertEqual([r["valid"] for r in rolls], [t["actions"][0]["valid"] for t in trs])
        self.assertEqual([r["reward"] for r in rolls], [t["question_records"][0]["qa_score"] for t in trs])
        rewards = torch.tensor([r["reward"] for r in rolls])
        expected = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        torch.testing.assert_close(expected, torch.tensor(args._last_update["groups"][0]["advantages"]))
        self.assertTrue(result["did_step"])
        self.assertAlmostEqual(result["loss"], captured["loss"], places=7)
        torch.testing.assert_close(a.model.weight.grad, b.model.weight.grad, rtol=0, atol=1e-7)
        torch.testing.assert_close(a.model.weight, b.model.weight, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
