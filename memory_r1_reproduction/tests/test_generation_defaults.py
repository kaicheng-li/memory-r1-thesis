"""Real Transformers generation must neutralize non-neutral checkpoint defaults."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_manager_grpo import TextModel
from scripts.train_answer_grpo import generate_completions
from scripts.construct_memory_bank import Manager
from scripts.generate_memory_augmented_answers import AnswerAgent
from evaluation.audit_checkpoints import AuditModel
from memfactory.modules.memory_extractor import NaiveExtractor
from memfactory.modules.memory_updater import NaiveUpdater
from memfactory.modules.memory_retriever import RerankRetriever


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, *args, **kwargs):
        return Batch(input_ids=torch.tensor([[4, 4, 4]]), attention_mask=torch.ones((1, 3), dtype=torch.long))

    def decode(self, *args, **kwargs):
        return '{"memory_list": []}'


class GenerationDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_positions=16, n_embd=16, n_layer=1, n_head=2,
            eos_token_id=2, pad_token_id=0, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
        self.model.eval()
        config = self.model.generation_config
        config.do_sample = True
        config.temperature, config.top_k, config.top_p, config.typical_p = 0.7, 1, 0.5, 0.6
        config.repetition_penalty, config.no_repeat_ngram_size = 1.05, 2
        self.tokenizer = Tokenizer()

    def wrapper(self, cls):
        obj = cls.__new__(cls)
        obj.model, obj.tokenizer, obj.device, obj.torch = self.model, self.tokenizer, "cpu", torch
        obj.max_new_tokens, obj.max_prompt_tokens, obj.max_generate_length = 2, 32, 2
        return obj

    def assert_raw_distribution(self, call):
        original = self.model.generate
        before = self.model.generation_config.to_dict()
        captured = []

        def capture(**kwargs):
            result = original(**kwargs, return_dict_in_generate=True, output_scores=True)
            captured.append(result)
            return result.sequences

        with patch.object(self.model, "generate", side_effect=capture):
            call()
        self.assertEqual(self.model.generation_config.to_dict(), before)
        self.assertEqual(len(captured), 1)
        self.model.eval()
        with torch.no_grad():
            result = captured[0]
            for step, scores in enumerate(result.scores):
                raw = self.model(result.sequences[:, :3+step]).logits[:, -1, :]
                torch.testing.assert_close(scores.softmax(-1), raw.softmax(-1), rtol=1e-5, atol=1e-6)

    def test_manager_answer_and_precheck_sampling_and_greedy(self):
        for temperature in (1.0, 0.0):
            for name, call in (
                ("manager", lambda: self.wrapper(TextModel).generate("p", 2, temperature, 32)),
                ("answer", lambda: generate_completions(self.model, self.tokenizer, "p", 1, 2, temperature, "cpu", 32)),
                ("precheck", lambda: self.wrapper(AuditModel).generate("p", 32, 2, temperature, 42)),
            ):
                with self.subTest(entry=name, temperature=temperature):
                    self.assert_raw_distribution(call)

    def test_deployment_and_data_builder_greedy(self):
        for name, call in (
            ("extract", lambda: self.wrapper(Manager).extract({"speaker": "A", "text": "x", "timestamp": "now"})),
            ("manager", lambda: self.wrapper(Manager).generate_prompt("p")),
            ("answer", lambda: self.wrapper(AnswerAgent).answer("q", [], ["A"])),
        ):
            with self.subTest(entry=name):
                self.assert_raw_distribution(call)

    def test_legacy_module_sampling(self):
        for cls in (NaiveExtractor, NaiveUpdater, RerankRetriever):
            with self.subTest(entry=cls.__name__):
                self.assert_raw_distribution(lambda: self.wrapper(cls)._generate_with_pytorch(
                    self.model, torch.tensor([[4, 4, 4]]), torch.ones((1, 3), dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
