from __future__ import annotations
import copy
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memfactory.oracle_reward import (estimate_advantages, safe_answer, ContextBudgetError,
                                      evaluate_controls, chat_token_count, extract_answer)
from memfactory.training_state import (capture_rng, restore_rng, preserve_rng,
                                       save_training_checkpoint, read_training_state)
from scripts.train_manager_grpo import train, build_parser, coverage_rewards, train_from_tuples
from evaluation.metrics import f1_score


class AdvantageTests(unittest.TestCase):
    def test_three_noop_examples(self):
        for values, baseline, expected, skip in [([.1]*3, .9, [-.8]*3, False),
                                                ([.2]*3, .2, [0]*3, True),
                                                ([0,.5,1], .2, [-.2,.3,.8], False)]:
            actual, skipped = estimate_advantages(torch.tensor(values), "noop", baseline)
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32))
            self.assertEqual(skipped, skip)

    def test_grpo_and_fixed_scale_clip(self):
        _, skip = estimate_advantages(torch.tensor([.1,.1]), "grpo")
        self.assertTrue(skip)
        values, skip = estimate_advantages(torch.tensor([0.,1.]), "grpo")
        torch.testing.assert_close(values, torch.tensor([-1.,1.]), atol=3e-6, rtol=0)
        values, _ = estimate_advantages(torch.tensor([-10.,10.]), "noop", scale=2, clip=3)
        self.assertEqual(values.tolist(), [-3,3])
        with self.assertRaises(ValueError):
            estimate_advantages(torch.tensor([1.]), "noop", scale=0)

    def test_terminal_mode_equal_total_different_credit(self):
        dense = coverage_rewards([.2,.9,.5])
        terminal = coverage_rewards([.2,.9,.5], mode="terminal")
        self.assertAlmostEqual(sum(dense), sum(terminal))
        self.assertNotEqual(dense, terminal)
        with self.assertRaises(ValueError):
            coverage_rewards([1.], gamma=.9, mode="terminal")

    def test_baseline_metric_and_answer_parser(self):
        self.assertEqual(f1_score("yes yes", "yes yes"), .5)
        self.assertEqual(extract_answer("reason <answer>Paris</answer>"), "Paris")
        self.assertEqual(extract_answer("**Answer:** Paris"), "Paris")


class CharacterTokenizer:
    eos_token_id = 2
    model_max_length = 100000
    def apply_chat_template(self, messages, **kwargs):
        return "CHAT:" + messages[0]["content"] + ":ASSISTANT"
    def __call__(self, text, **kwargs):
        assert kwargs.get("truncation") is False
        return {"input_ids": list(range(len(text)))}


class Reader:
    def __init__(self, budget=4500):
        self.tokenizer = CharacterTokenizer()
        self.max_prompt_tokens = budget
        self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=20000))
        self.prompts = []
    def generate(self, prompt, max_new_tokens, temperature):
        self.prompts.append(prompt)
        return ("Ada: Paris" if prompt.startswith("Extract concise") else "Answer: Paris"), torch.tensor([7,2])


class ContextTests(unittest.TestCase):
    def test_direct_and_explicit_overflow(self):
        reader = Reader(10000)
        value = safe_answer(reader, "Where?", [{"speaker":"Ada", "text":"Paris"}], ["Ada"], 50)
        self.assertEqual(value["status"], "direct")
        self.assertFalse(value["input_truncated"])
        reader.max_prompt_tokens = 20
        with self.assertRaises(ContextBudgetError):
            safe_answer(reader, "Where?", [], ["Ada"], 50, "error")
        self.assertEqual(len(reader.prompts), 1)

    def test_compression_reads_all_source_characters_including_long_fact(self):
        reader = Reader()
        original = "".join(f"FACT-{i:04d};" for i in range(1200))
        result = safe_answer(reader, "Where?", [{"speaker":"Ada", "text":original}], ["Ada"], 50, "compress")
        extracts = [p for p in reader.prompts if p.startswith("Extract concise")]
        # First pass fragments cover the entire source, in order, without dropped tails.
        fragments = [row["text"] for p in extracts for row in json.loads(p.split("Records: ",1)[1])]
        self.assertEqual("".join(fragments), original)
        self.assertEqual(result["status"], "compressed")
        self.assertTrue(all(c["prompt_tokens"] <= c["budget"] for c in result["calls"]))

    def test_output_budget_failure_is_not_a_wrong_answer(self):
        reader = Reader(10000)
        with patch.object(reader, "generate", return_value=("unfinished", torch.tensor([1]*5))):
            with self.assertRaisesRegex(ContextBudgetError, "output limit"):
                safe_answer(reader, "Where?", [], ["Ada"], 5, "error")

    def test_gate_and_context_errors_separate(self):
        item = {"participants":["Ada"], "dialogue_turns":[{"speaker":"Ada", "dia_id":"D1:1",
                "facts":[{"text":"Paris"}]}], "linked_questions":[
                {"question_id":"1", "question":"q1", "answer":"Paris"},
                {"question_id":"2", "question":"q2", "answer":"Paris"}]}
        args = SimpleNamespace(answer_max_new_tokens=50, control_gate="threshold", control_threshold=.5)
        def answer(_model, q, memories, *_args):
            return {"prediction": "Paris" if q == "q1" and memories else "wrong", "status":"direct"}
        with patch("memfactory.oracle_reward.safe_answer", side_effect=answer) as mock:
            values = evaluate_controls(None, item, args, f1_score)
        self.assertEqual(mock.call_count, 4)
        self.assertEqual([v["eligible"] for v in values], [True,False])
        self.assertEqual(values[1]["exclusion_reason"], "below_control_threshold")
        with patch("memfactory.oracle_reward.safe_answer", side_effect=ContextBudgetError("overflow")):
            values = evaluate_controls(None, item, args, f1_score)
        self.assertEqual(values[0]["exclusion_reason"], "context_error")


class SavedLinear(torch.nn.Linear):
    def save_pretrained(self, directory):
        torch.save(self.state_dict(), Path(directory)/"adapter.pt")


class TinyPolicy:
    def __init__(self, path, device, trainable, adapter_path=None, max_prompt_tokens=4096):
        self.device = "cpu"
        self.calls = 0
        self.model = SavedLinear(1,1,bias=False)
        if adapter_path:
            self.model.load_state_dict(torch.load(Path(adapter_path)/"adapter.pt", weights_only=True))
        self.tokenizer = SimpleNamespace(save_pretrained=lambda directory: None)
    def generate(self, *_args):
        token = 1 + int(torch.randint(0,3,()).item()) + int(random.random()>.5) + int(np.random.random()>.5)
        self.calls += 1
        return json.dumps({"memory": [{"event": "ADD", "text": f"memory{self.calls % 2}"}]}), torch.tensor([token])
    def log_probability(self, _prompt, tokens):
        return self.model(tokens.float().reshape(-1,1)).flatten()


def sample_row(index=0):
    return {"dialogue_id":"conv-26", "split":"train", "turn_index":index,
            "participants":["Ada"], "temporal_memory_bank":[], "current_turn":{},
            "dialogue_turns":[{"dia_id":"D1:1", "speaker":"Ada", "facts":[{"text":"Paris"}]}],
            "linked_questions":[{"question_id":"1", "question":"Where?", "answer":"Paris", "evidence":["D1:1"]}]}


def fake_answer(_model, _q, memories, *_args):
    teacher = memories and str(memories[0].get("id", "")).startswith("teacher:")
    return {"prediction":"Paris" if not memories or teacher or memories[-1]["text"] == "memory1" else "wrong", "status":"direct"}


class IntegrationTests(unittest.TestCase):
    def test_resume_matches_continuous_optimizer_rng_and_cursor(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"MEMORY_RETRIEVAL_BACKEND":"lexical"}), \
             patch("scripts.train_manager_grpo.TextModel", TinyPolicy), \
             patch("memfactory.oracle_reward.safe_answer", side_effect=fake_answer), \
             patch("scripts.train_manager_grpo.safe_answer", side_effect=fake_answer):
            root = Path(temp)
            source = root/"tuples.jsonl"
            source.write_text("\n".join(json.dumps(sample_row(i)) for i in range(2)),encoding="utf-8")
            def args(output, resume=None):
                values = ["--data-path",str(source),"--manager-model","fake","--answer-model","fake",
                          "--output-dir",str(root/output),"--epochs","1", "--beta","0",
                          "--evidence-weight","0", "--num-generations","4", "--checkpoint-every","1"]
                if resume:
                    values += ["--resume",str(resume)]
                return build_parser().parse_args(values)
            train(args("continuous"))
            checkpoint = root/"interrupted"/"epoch_0_tuple_1"
            def interrupt(*a, **kw):
                save_training_checkpoint(*a, **kw)
                raise RuntimeError("simulated interruption")
            with patch("memfactory.training_state.save_training_checkpoint",side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train(args("interrupted"))
            train(args("resumed", checkpoint))
            direct = torch.load(root/"continuous"/"epoch_1"/"training_state.pt",weights_only=False)
            resumed = torch.load(root/"resumed"/"epoch_1"/"training_state.pt",weights_only=False)
            w1 = torch.load(root/"continuous"/"epoch_1"/"adapter.pt",weights_only=True)
            w2 = torch.load(root/"resumed"/"epoch_1"/"adapter.pt",weights_only=True)
            torch.testing.assert_close(w1["weight"],w2["weight"],rtol=0,atol=0)
            for key,value in direct["optimizer"]["state"][0].items():
                torch.testing.assert_close(value,resumed["optimizer"]["state"][0][key],rtol=0,atol=0)
            torch.testing.assert_close(direct["rng"]["torch"],resumed["rng"]["torch"])
            self.assertEqual(direct["rng"]["python"],resumed["rng"]["python"])
            np.testing.assert_equal(direct["rng"]["numpy"],resumed["rng"]["numpy"])
            logs = [json.loads(line) for line in (root/"resumed"/"groups.jsonl").read_text().splitlines()]
            self.assertEqual([r["tuple_index"] for r in logs if "tuple_index" in r], [1])
            self.assertEqual(logs[-1]["update"]["eligible_actions"], 4)
            self.assertAlmostEqual(sum(logs[-1]["update"]["groups"][0]["advantages"]), 0.0)
            self.assertEqual(direct["scheduler"], resumed["scheduler"])
            self.assertEqual(direct["epoch_stats"], resumed["epoch_stats"])
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                read_training_state(checkpoint, {})

    def test_preserve_rng_restores_after_exception(self):
        before = capture_rng()
        with self.assertRaises(RuntimeError):
            with preserve_rng():
                random.random(); np.random.random(); torch.rand(1)
                raise RuntimeError()
        after = capture_rng()
        self.assertEqual(before["python"],after["python"])
        np.testing.assert_equal(before["numpy"],after["numpy"])
        torch.testing.assert_close(before["torch"],after["torch"])

    def test_diagnostics_do_not_mask_qa_or_potential(self):
        row = sample_row()
        row["linked_questions"].append({"question_id":"2","question":"excluded", "answer":"bad","evidence":["D1:9"]})
        args = build_parser().parse_args(["--data-path","unused","--manager-model","fake","--answer-model","fake",
                                         "--control-diagnostics"])
        controls = [{"eligible":True,"noop":{"score":.2}}, {"eligible":False,"noop":{"score":.9}}]
        seen = []
        def phi(q,*_a):
            seen.append(q["question_id"])
            return .4
        with patch.dict(os.environ,{"MEMORY_RETRIEVAL_BACKEND":"lexical"}), \
             patch("scripts.train_manager_grpo.evaluate_controls",return_value=controls) as control, \
             patch("scripts.train_manager_grpo.evidence_potential",side_effect=phi), \
             patch("scripts.train_manager_grpo.safe_answer",return_value={"prediction":"Paris"}), \
             patch("scripts.train_manager_grpo.apply_policy_update",return_value=float("nan")) as update:
            train_from_tuples(args,TinyPolicy("","cpu",True),None,TinyPolicy("","cpu",False),None,f1_score,[row])
        self.assertEqual(control.call_count,1)
        self.assertEqual(seen,["1", "2"]*4)
        self.assertNotIn("noop_baseline", update.call_args.args[3][0])
        self.assertEqual(len(args._question_rewards), 2)
        self.assertTrue(all(abs(r[0]-1.2)<1e-6 for r in args._question_rewards[0]))
        self.assertTrue(all(abs(r[0]-.2)<1e-6 for r in args._question_rewards[1]))


class PipelineTests(unittest.TestCase):
    def test_minilm_dispatch_and_stable_embedding_ranking(self):
        from memfactory.memory_runtime import retrieve
        vectors = {"query": np.array([1.,0.]), "first": np.array([0.,1.]),
                   "second": np.array([1.,0.]), "third": np.array([1.,0.])}
        bank = [{"text":word} for word in ("first","second","third")]
        with patch.dict(os.environ,{"MEMORY_RETRIEVAL_BACKEND":"minilm"}), \
             patch("memfactory.retrieval.embedding", side_effect=lambda text,*_: vectors[text]):
            self.assertEqual([r["text"] for r in retrieve("query",bank,3)], ["second","third","first"])
            self.assertEqual(retrieve("query",bank,0),[])

    def test_precheck_controls_match_training_and_gold_is_distinct(self):
        from scripts.precheck_sequential import precheck_item
        item = sample_row()
        args = SimpleNamespace(answer_max_new_tokens=50, control_gate="threshold",control_threshold=.5,
                               context_overflow="compress", compression_tokens=30, manager_top_k=5,
                               answer_top_k=30, max_new_tokens=10, include_gold_support=True, include_raw=False)
        with patch.dict(os.environ,{"MEMORY_RETRIEVAL_BACKEND":"lexical"}), \
             patch("memfactory.oracle_reward.safe_answer",side_effect=fake_answer), \
             patch("scripts.precheck_sequential.safe_answer",side_effect=fake_answer):
            train_controls = evaluate_controls(None,item,args,f1_score)
            rows = precheck_item(item,TinyPolicy("","cpu",False),None,args,f1_score)
        self.assertEqual(rows[0]["conditions"]["teacher_all_facts"],train_controls[0]["teacher"])
        self.assertEqual(rows[0]["conditions"]["empty_noop"],train_controls[0]["noop"])
        self.assertEqual(len(rows[0]["conditions"]),4)
        self.assertEqual(rows[0]["eligible"],train_controls[0]["eligible"])

    def test_full_test_evaluation_uses_shared_runtime_without_gate(self):
        from evaluation.evaluate_test_final import evaluate_dialogues, merge
        item = sample_row()
        timeline = {"dialogue_id":"conv-41", "participants":item["participants"],
                    "turns":item["dialogue_turns"],
                    "questions":[dict(item["linked_questions"][0],category=1)]}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{"MEMORY_RETRIEVAL_BACKEND":"lexical"}):
            path=Path(temp)/"test.json"
            path.write_text(json.dumps([timeline]),encoding="utf-8")
            args=SimpleNamespace(model_path="fake",manager_adapter="unused",answer_adapter="unused",
                                 manager_output_dir="",answer_output_dir="",device="cpu",seed=42,
                                 max_prompt_tokens=4096,max_new_tokens=10,answer_max_new_tokens=50,
                                 answer_top_k_per_speaker=30,manager_top_k=5,timelines=str(path),dialogues=None,
                                 output_dir=str(Path(temp)/"out"),force=False,context_overflow="compress",
                                 compression_tokens=30,judge_base_url=None)
            def factory(*a,**kw):
                return TinyPolicy("","cpu",False)
            with patch("scripts.train_manager_grpo.TextModel",side_effect=factory), \
                 patch("memfactory.oracle_reward.safe_answer",side_effect=fake_answer):
                evaluate_dialogues(args)
            merge(args)
            report=json.loads((Path(args.output_dir)/"final_summary.json").read_text())
            self.assertEqual(report["overall"]["n"],1)
            self.assertIn("token_f1_multiset",report["overall"])

    def test_speaker_attribution_survives_chunking(self):
        reader=Reader(4500)
        memories=[{"speaker":name,"text":"long fact "*1000} for name in ("Ada","Bob")]
        safe_answer(reader,"Where?",memories,["Ada","Bob"],50,"compress")
        final=reader.prompts[-1]
        self.assertIn('Memories for user Bob:',final)
        self.assertIn('Memories for user Ada:',final)
        for prompt in reader.prompts[:-1]:
            rows=json.loads(prompt.split("Records: ",1)[1])
            self.assertEqual(len({r["speaker"] for r in rows}),1)


if __name__ == "__main__":
    unittest.main()
