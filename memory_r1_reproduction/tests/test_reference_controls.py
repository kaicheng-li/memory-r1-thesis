import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memfactory.reference_advantage import advantages
from memfactory.reference_answer import (RawReferences, ReferenceCache, reference_prompt, digest,
    measure_budgets, protocol_for, local_model_identity)
from memfactory.reference_reporting import reference_report
from scripts import train_manager_reference as trainer
from test_reference_baseline import Policy


def fixture():
    turns = [{"turn_index":i,"dia_id":f"d{i}","speaker":"A","timestamp":"today",
              "text":f"raw original {i}","facts":[{"text":"DO NOT READ FACTS"}]} for i in range(3)]
    q = {"question_id":"0","question":"Where?","answer":"Paris"}
    timeline = {"dialogue_id":"conv-26","split":"train","turns":turns,"questions":[q]}
    state = {**q,"dialogue_id":"conv-26","split":"train","target_turn":1,
             "old_bank":[{"source_turn":0}], "current_facts":[{"text":"Paris"}],"participants":["A"]}
    return timeline, state


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "fake-template"
    special_tokens_map = {"eos_token":"EOS"}
    model_max_length = 32768
    def apply_chat_template(self, messages, **kwargs):
        return "USER " + messages[0]["content"] + " ASSISTANT"
    def __call__(self, text, **kwargs):
        return {"input_ids":[1]*len(text.split())}
    def decode(self, ids, **kwargs):
        return "**Answer:** Paris"


class Batch(dict):
    def to(self, _):
        return self


class ReferenceAdvantageTests(unittest.TestCase):
    def test_four_cases_and_illegal_minus_reference(self):
        r = torch.tensor([0., 1., .25, .75])
        a, signal, _ = advantages(r, "raw-reference", reference=1.)
        torch.testing.assert_close(a, torch.tensor([-1., 0., -.75, -.25]))
        self.assertTrue(signal)
        self.assertEqual(float(a[0]), -1.)  # illegal r=0 -> -c, not hidden/dropped
        a, _, _ = advantages(r, "raw-reference", reference=.6)
        self.assertAlmostEqual(float(a[0]), -.6, places=6)
        a, _, _ = advantages(r, "raw-reference", reference=0.)
        torch.testing.assert_close(a, advantages(r, "fixed-zero")[0])

    def test_all_negative_equal_group_updates_and_equal_reference_skips(self):
        r = torch.full((4,), .25)
        a, signal, _ = advantages(r, "raw-reference", reference=1.)
        self.assertTrue(signal)
        torch.testing.assert_close(a, torch.full((4,), -.75))
        self.assertFalse(advantages(r, "grpo")[1])
        self.assertFalse(advantages(r, "raw-reference", reference=.25)[1])

    def test_change_reference_changes_actual_gradient(self):
        args = SimpleNamespace(advantage_mode="raw-reference", advantage_scale=1., variance_epsilon=1e-12,
                               beta=0., grad_clip_norm=100., max_prompt_tokens=32, clip_epsilon=.2)
        gradients = []
        for c in (0., .8):
            p = Policy([])
            args._reference_score = c
            rows = [{"prompt":"p","completion_ids":torch.tensor([i+1]),"reward":.25,"valid":i != 0}
                    for i in range(4)]
            optimizer = torch.optim.SGD(p.model.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
            result = trainer.update_policy(p, optimizer, scheduler, rows, args, 1)
            self.assertTrue(result["did_step"])
            gradients.append(float(p.model.weight.grad))
        self.assertNotEqual(gradients[0], gradients[1])

    def test_no_recenter_after_subtraction(self):
        r = torch.tensor([0., .25, .75, 1.])
        torch.testing.assert_close(advantages(r,"raw-reference",.5)[0],r-.5)
        def z(v):
            return (v-v.mean())/v.std(unbiased=False)
        torch.testing.assert_close(z(r), z(1.5*r-.25))


class ReferenceCacheTests(unittest.TestCase):
    def test_causal_prefix_and_no_gold_or_facts_in_prompt(self):
        timeline, state = fixture()
        refs = RawReferences([timeline],[state])
        prompt = reference_prompt(refs.entry(state))
        self.assertIn("raw original 1",prompt)
        for forbidden in ("raw original 2", "DO NOT READ FACTS", "Paris"):
            self.assertNotIn(forbidden,prompt)
        for change in (lambda s:s["old_bank"][0].update(source_turn=1), lambda s:s.update(question="wrong")):
            other=copy.deepcopy(state);change(other)
            with self.assertRaises(ValueError):RawReferences([timeline],[other])

    def make_reader(self):
        return SimpleNamespace(tokenizer=Tokenizer(), device="cpu", model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=32768,rope_scaling=None)))

    def test_budget_30720_plus_2048_and_overflow(self):
        args=SimpleNamespace(reference_max_prompt_tokens=30720,answer_max_new_tokens=2048)
        reader=self.make_reader()
        self.assertEqual(protocol_for(reader,args,{"weights":"hash"})["prompt_limit"],30720)
        args.reference_max_prompt_tokens=32768
        with self.assertRaises(ValueError):protocol_for(reader,args,{})
        timeline,state=fixture()
        self.assertEqual(len(measure_budgets(RawReferences([timeline],[state]),reader.tokenizer,1)["over_budget"]),1)

    def test_native_rope_metadata_is_not_context_extension(self):
        from transformers import Qwen2Config
        from memfactory.reference_answer import validate_native_rope
        args=SimpleNamespace(reference_max_prompt_tokens=30720,answer_max_new_tokens=2048)
        reader=self.make_reader()
        reader.model.config=Qwen2Config(max_position_embeddings=32768,rope_theta=1000000.)
        before=copy.deepcopy(reader.model.config.to_dict())
        self.assertEqual(protocol_for(reader,args,{})["capacity"],32768)
        self.assertEqual(reader.model.config.to_dict(),before)
        for field in ("rope_scaling", "rope_parameters"):
            for value in (None, {}, {"rope_type":"default","rope_theta":1000000.},
                          {"type":"default","factor":1.}):
                with self.subTest(field=field,value=value):
                    validate_native_rope(SimpleNamespace(**{field:value}))
            for value in ({"rope_type":"yarn","factor":4.}, {"type":"linear","factor":2.},
                          {"rope_type":"default","factor":4.}, {"factor":4.},
                          {"rope_type":"default","type":"yarn"}):
                with self.subTest(field=field,value=value):
                    with self.assertRaisesRegex(ValueError,"does not enable"):
                        validate_native_rope(SimpleNamespace(**{field:value}))

    def test_cache_rng_failure_and_hashes(self):
        timeline,state=fixture()
        refs=RawReferences([timeline],[state]);entry=refs.entry(state)
        reader=self.make_reader()
        protocol={"prompt_limit":30720,"output_limit":3,"model":"a","prompt":"p"}
        from memfactory.reference_answer import count_tokens
        n=count_tokens(reader.tokenizer,reference_prompt(entry))
        inputs=Batch(input_ids=torch.ones((1,n),dtype=torch.long),attention_mask=torch.ones((1,n),dtype=torch.long))
        calls=[]
        def generate(*_):
            calls.append(1)
            torch.rand(9)
            return torch.cat([inputs["input_ids"],torch.tensor([[5,2]])],dim=1)
        with tempfile.TemporaryDirectory() as directory, \
             patch("memfactory.reference_answer.tokenize_chat_prompt",return_value=inputs), \
             patch("memfactory.reference_answer.generate_policy",side_effect=generate):
            cache=ReferenceCache(directory,protocol)
            before=torch.get_rng_state().clone()
            first=cache.get(entry,reader,epoch=0)
            torch.testing.assert_close(before,torch.get_rng_state())
            second=cache.get(entry,reader,epoch=2)
            self.assertFalse(first["cache_hit"]);self.assertTrue(second["cache_hit"])
            self.assertEqual(second["generated_epoch"],0);self.assertEqual(second["access_epoch"],2)
            self.assertEqual(len(calls),1);self.assertEqual(first["f1"],1.)
            self.assertFalse(first["input_truncated"])
            self.assertIsNone(ReferenceCache(directory,{**protocol,"model":"changed"}).read(entry))
            self.assertIsNone(cache.read({**entry,"gold":"other"}))
            report=reference_report(cache,refs,Path(directory)/"report.json")
            self.assertIsNone(report["empirical_all_negative_fraction"])
            bad=json.loads(cache.path(entry).read_text());bad["input_truncated"]=True
            cache.path(entry).write_text(json.dumps(bad))
            with self.assertRaises(ValueError):cache.read(entry)
        with tempfile.TemporaryDirectory() as directory, \
             patch("memfactory.reference_answer.tokenize_chat_prompt",return_value=inputs), \
             patch("memfactory.reference_answer.generate_policy",return_value=torch.cat([inputs["input_ids"],torch.tensor([[4,4,4]])],dim=1)):
            cache=ReferenceCache(directory,protocol)
            with self.assertRaisesRegex(ValueError,"length limit"):cache.get(entry,reader)
            self.assertFalse(cache.path(entry).exists())

    def test_model_fingerprint_detects_in_place_weight_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.safetensors";path.write_bytes(b"a")
            before=local_model_identity(directory)
            path.write_bytes(b"b")
            self.assertNotEqual(before,local_model_identity(directory))


if __name__ == "__main__":unittest.main()
