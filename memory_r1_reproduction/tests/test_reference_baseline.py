"""Opening gate: independently load the pinned baseline and compare the port."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_manager_reference as port

PIN = "87f13c579c3121fe9a76119ac5a15b059681954d"


def pinned_module():
    def source(path):
        return subprocess.check_output(["git", "show", f"{PIN}:memory_r1_reproduction/{path}"],
                                       cwd=ROOT, encoding="utf-8")
    module = types.ModuleType("reference_pinned_baseline")
    module.__file__ = str(ROOT / "scripts/train_manager_grpo.py")
    sys.modules[module.__name__] = module
    exec(source("scripts/train_manager_grpo.py"), module.__dict__)
    env = types.ModuleType("reference_pinned_env")
    sys.modules[env.__name__] = env
    exec(source("memfactory/manager_env.py"), env.__dict__)
    module.validate_and_apply = env.validate_and_apply
    metrics = types.ModuleType("reference_pinned_metrics")
    exec(source("evaluation/metrics.py"), metrics.__dict__)
    module.f1_score, module.exact_match = metrics.f1_score, metrics.exact_match
    for path, name in [("memfactory/modules/memory_updater.py", "build_manager_input"),
                       ("memfactory/modules/memory_retriever.py", "build_answer_input"),
                       ("scripts/construct_memory_bank.py", "parse_manager_output")]:
        text = source(path)
        tree = ast.parse(text)
        if name == "build_answer_input":
            # The source also defines dependencies; extract the canonical prompt.
            assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "RERANK_PROMPT" for t in n.targets))
            module.RERANK_PROMPT = ast.literal_eval(assign.value)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {"json": json, "RERANK_PROMPT": getattr(module, "RERANK_PROMPT", "")}
        exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
        setattr(module, name, namespace[name])
    return module


class Policy:
    device = "cpu"
    def __init__(self, outputs):
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.model.weight, .1)
        self.model.disable_adapter = self.disable_adapter
        self.reference_mode = False
        self.outputs, self.index, self.prompts = outputs, 0, []

    @contextmanager
    def disable_adapter(self):
        self.reference_mode = True
        try:
            yield
        finally:
            self.reference_mode = False

    def generate(self, prompt, *_):
        self.prompts.append(prompt)
        raw = self.outputs[self.index]
        self.index += 1
        return raw, torch.tensor([self.index])

    def log_probability(self, prompt, ids, *_):
        return torch.zeros(len(ids)) if self.reference_mode else self.model(ids.float().reshape(-1, 1)).flatten()


class ReferenceBaselineGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = pinned_module()

    def test_pinned_transfer_score_advantage_loss_gradient_update(self):
        outputs = ['{"memory":[{"event":"ADD","text":"Paris"}]}', 'bad json',
                   '{"memory":[]}', '{"memory":[{"event":"UPDATE","id":"d:m0","text":"Paris"}]}']
        state = dict(dialogue_id="d", question_id="q", question="Where?", answer="Paris", participants=["A"],
                     target_turn=1, target_speaker="A", target_timestamp="now", current_facts=[{"text":"Paris"}],
                     old_bank=[{"id":"d:m0", "text":"Rome", "speaker":"A", "timestamp":"then", "source_turn":0}])
        for beta in (0., .01):
            with self.subTest(beta=beta):
                args = SimpleNamespace(manager_top_k=5, answer_top_k_per_speaker=30, num_generations=4,
                    max_new_tokens=2048, max_prompt_tokens=4096, answer_max_new_tokens=2048,
                    temperature=1., variance_epsilon=1e-12, clip_epsilon=.2, beta=beta,
                    grad_clip_norm=1., advantage_mode="grpo", advantage_scale=1.)
                observed = []
                def answer(_a, _q, bank, *_):
                    observed.append(copy.deepcopy(bank))
                    return "Paris" if any(m["text"] == "Paris" for m in bank) else "Rome"
                a, b = Policy(outputs), Policy(outputs)
                opt_a, opt_b = [torch.optim.SGD(p.model.parameters(), lr=.1) for p in (a, b)]
                sa, sb = [torch.optim.lr_scheduler.LambdaLR(o, lambda _:1.) for o in (opt_a, opt_b)]
                with patch.dict(os.environ, {"MEMORY_RETRIEVAL_BACKEND":"lexical"}), \
                     patch.object(self.baseline, "answer_from_bank", side_effect=answer), \
                     patch.object(port, "answer_from_bank", side_effect=answer):
                    ra = self.baseline.rollout_group(a, object(), state, args)
                    rb = port.rollout_group(b, object(), state, args)
                    ua = self.baseline.update_policy(a, opt_a, sa, ra, args, 1)
                    ub = port.update_policy(b, opt_b, sb, rb, args, 1)
                self.assertEqual(a.prompts, b.prompts)
                self.assertEqual(observed[:3], observed[3:])
                self.assertEqual([x["reward"] for x in ra], [x["reward"] for x in rb])
                self.assertEqual([x["valid"] for x in ra], [x["valid"] for x in rb])
                rewards = torch.tensor([x["reward"] for x in ra])
                expected_advantages = (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
                torch.testing.assert_close(torch.tensor([x["advantage"] for x in rb]), expected_advantages)
                self.assertEqual(state["old_bank"][0]["text"], "Rome")
                self.assertAlmostEqual(ua["loss"], ub["loss"], places=7)
                self.assertAlmostEqual(ua["reward_std"], ub["reward_std"], places=7)
                torch.testing.assert_close(a.model.weight.grad, b.model.weight.grad, atol=1e-7, rtol=0)
                torch.testing.assert_close(a.model.weight, b.model.weight, atol=1e-7, rtol=0)
                self.assertEqual(sa.state_dict(), sb.state_dict())

    def test_answer_prompt_and_metrics_match_pin(self):
        from memfactory.modules.memory_retriever import build_answer_input
        memories = {"A":[{"timestamp":"today", "text":"Paris"}], "B":[]}
        self.assertEqual(build_answer_input("Where?", memories), self.baseline.build_answer_input("Where?", memories))
        for predicted, gold in [("Paris", "Paris"), ("yes yes extra", "yes yes"), ("", "x")]:
            self.assertEqual(port.f1_score(predicted, gold), self.baseline.f1_score(predicted, gold))


if __name__ == "__main__":
    unittest.main()
