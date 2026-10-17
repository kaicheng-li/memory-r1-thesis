"""Exercise real Transformers generation with non-neutral checkpoint defaults."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_manager_grpo import TextModel


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids.tolist()))


class ManagerGenerationTests(unittest.TestCase):
    def test_checkpoint_penalties_do_not_change_rollout_distribution(self):
        # Tiny random model: no downloaded weights, CUDA, or model services.
        reader = TextModel.__new__(TextModel)
        reader.device = "cpu"
        reader.max_prompt_tokens = 32
        reader.tokenizer = Tokenizer()
        reader.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_positions=16, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        ))
        config = reader.model.generation_config
        config.do_sample = True
        config.temperature = 0.7
        config.top_k = 3
        config.top_p = 0.8
        config.typical_p = 0.9
        config.repetition_penalty = 1.05
        config.no_repeat_ngram_size = 2
        before = config.to_dict()
        # A repeated prefix ensures n-gram blocking would actually take effect.
        inputs = Batch(input_ids=torch.tensor([[4, 4, 4]]),
                       attention_mask=torch.ones((1, 3), dtype=torch.long))
        generate = reader.model.generate
        results = []

        def capture(**kwargs):
            result = generate(**kwargs, return_dict_in_generate=True, output_scores=True)
            results.append(result)
            return result.sequences

        for temperature in (1.0, 0.0):
            with self.subTest(temperature=temperature):
                reader.model.train()
                with patch("scripts.train_manager_grpo.tokenize_chat_prompt", return_value=inputs), \
                     patch.object(reader.model, "generate", side_effect=capture):
                    _, ids = reader.generate("prompt", 2, temperature)
                self.assertTrue(reader.model.training)
                self.assertEqual(config.to_dict(), before)
                result = results[-1]
                torch.testing.assert_close(ids, result.sequences[0, 3:])
                reader.model.eval()
                with torch.no_grad():
                    for step, scores in enumerate(result.scores):
                        prefix = result.sequences[:, :3 + step]
                        raw = reader.model(input_ids=prefix).logits[:, -1, :]
                        # generate() scores include its actual logits processors
                        # and warpers. Compare against the policy being scored.
                        torch.testing.assert_close(scores.softmax(-1), raw.softmax(-1),
                                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
