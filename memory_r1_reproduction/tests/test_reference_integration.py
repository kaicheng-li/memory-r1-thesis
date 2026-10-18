"""Exercise the real train loop/cache/checkpoints with CPU policies, no downloads."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_manager_reference as trainer
from memfactory.reference_answer import count_tokens
from test_reference_baseline import Policy
from test_reference_controls import Tokenizer, Batch


class ReferenceIntegrationTests(unittest.TestCase):
    def test_cache_training_reports_and_completed_resume_to_new_output(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = trainer.build_parser().parse_args([
                "--manager-model", directory, "--answer-model", directory,
                "--output-dir", str(Path(directory)/"run"),
                "--reference-cache-dir", str(Path(directory)/"cache"),
                "--advantage-mode", "raw-reference", "--epochs", "1", "--save-every-groups", "75"])
            for name in ("data_path", "validation_states", "validation_timelines", "reference_timelines"):
                setattr(args, name, str(ROOT/getattr(args,name)))
            accelerator = SimpleNamespace(device="cpu", num_processes=1, process_index=0,
                                          is_main_process=True, is_local_main_process=False)
            created = []
            def model(_path, _device, trainable, adapter_path=None):
                p = Policy([])
                p.tokenizer = Tokenizer()
                p.model.config = SimpleNamespace(max_position_embeddings=32768, rope_scaling=None)
                p.model.save_pretrained = lambda path: torch.save(p.model.state_dict(), Path(path)/"adapter.pt")
                p.tokenizer.save_pretrained = lambda path: None
                if adapter_path:
                    p.model.load_state_dict(torch.load(Path(adapter_path)/"adapter.pt", weights_only=True))
                if not trainable:
                    p.model.requires_grad_(False)
                created.append((trainable,p))
                return p
            def tokenize(tokenizer, prompt, limit):
                n = count_tokens(tokenizer,prompt)
                return Batch(input_ids=torch.ones((1,n),dtype=torch.long))
            def generate(_model, _tokenizer, inputs, *_):
                torch.rand(5)  # Must not perturb training's RNG on cache miss.
                return torch.cat([inputs["input_ids"],torch.tensor([[7,2]])],dim=1)
            def rollout(*_):
                return [{"prompt":"p", "completion_ids":torch.tensor([i+1]),
                         "reward":r, "em":float(r==1), "valid":bool(i), "actions":int(bool(i)),
                         "error":None if i else "parse", "prediction":"x", "raw_action":"x",
                         "completion_tokens":1, "elapsed_seconds":0.} for i,r in enumerate([0.,.2,.7,1.])]
            def validate(_manager, _answer, _args, epoch):
                torch.rand(5)
                metric = {"f1":.4 if epoch==0 else .3, "em":.2}
                return {"single_step":{"summary":metric}, "roll_forward":{"summary":metric}}
            for target, value in [("Accelerator",lambda:accelerator), ("TextModel",model),
                                  ("local_model_identity",lambda _: {"weights":"mock"}),
                                  ("rollout_group",rollout), ("validate_epoch",validate)]:
                stack.enter_context(patch.object(trainer,target,side_effect=value))
            stack.enter_context(patch("memfactory.reference_answer.tokenize_chat_prompt",side_effect=tokenize))
            generated = stack.enter_context(patch("memfactory.reference_answer.generate_policy",side_effect=generate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            trainer.train(args)
            output = Path(args.output_dir)
            groups = [json.loads(line) for line in (output/"groups.jsonl").read_text().splitlines()]
            self.assertEqual(len(groups),150)
            self.assertEqual(generated.call_count,150)
            for group in groups:
                self.assertFalse(group["reference"]["input_truncated"])
                self.assertTrue(group["reference"]["cache_hit"])
                self.assertEqual(group["reference"]["generated_epoch"],0)
                self.assertEqual(group["reference"]["access_epoch"],1)
                self.assertAlmostEqual(group["candidates"][0]["invalid_action_advantage"],
                                       -group["reference"]["f1"],places=6)
            self.assertIsNone(json.loads((output/"best_checkpoint.json").read_text())["adapter"])
            self.assertTrue((output/"reference_report.md").is_file())
            self.assertTrue((output/"checkpoints/epoch_1_group_75/trainer_state.pt").is_file())
            self.assertTrue(all(p.grad is None for flag,m in created if not flag for p in m.model.parameters()))
            args.resume_from = str(output/"epoch_1")
            args.output_dir = str(Path(directory)/"resumed")
            trainer.train(args)
            self.assertEqual(generated.call_count,150)
            self.assertFalse((Path(args.output_dir)/"groups.jsonl").exists())
            self.assertIsNone(json.loads((Path(args.output_dir)/"best_checkpoint.json").read_text())["adapter"])
            args.advantage_mode = "fixed-zero"
            with self.assertRaisesRegex(ValueError,"contract mismatch"):
                trainer.train(args)


if __name__ == "__main__":
    unittest.main()
