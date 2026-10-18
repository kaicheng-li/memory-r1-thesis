import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_manager_reference as trainer
from memfactory.reference_runtime import selection_record
from evaluation.evaluate_test_final import resolve_adapter
from test_reference_baseline import Policy


class ReferenceResumeTests(unittest.TestCase):
    def test_resume_restores_rng_optimizer_scheduler_and_cursor(self):
        accelerator=SimpleNamespace(is_main_process=True,num_processes=1,process_index=0,device="cpu")
        args=SimpleNamespace(advantage_mode="raw-reference",advantage_scale=1.,_reference_score=.6,
            variance_epsilon=1e-12,beta=0.,grad_clip_norm=100.,max_prompt_tokens=32,clip_epsilon=.2)
        def make():
            policy=Policy([])
            policy.reference_contract={"mode":"raw-reference","world_size":1,"hash":"fixed"}
            policy.model.save_pretrained=lambda directory:torch.save(policy.model.state_dict(),Path(directory)/"adapter.pt")
            policy.tokenizer=SimpleNamespace(save_pretrained=lambda directory:None)
            optimizer=torch.optim.AdamW(policy.model.parameters(),lr=.01,weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/10,1))
            return policy,optimizer,scheduler
        def step(p,o,s):
            rewards=torch.rand(4).tolist()
            rows=[{"prompt":"p","completion_ids":torch.tensor([i+1]),"reward":r,"valid":True}
                  for i,r in enumerate(rewards)]
            trainer.update_policy(p,o,s,rows,args,1)
            return rewards
        torch.manual_seed(42);random.seed(42)
        a,oa,sa=make();step(a,oa,sa)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"epoch_1_group_1"
            trainer.save_checkpoint(path,a,oa,sa,accelerator,
                {"resume_epoch":1,"next_group_index":1,"global_step":1})
            expected_rewards=step(a,oa,sa)
            b,ob,sb=make()
            b.model.load_state_dict(torch.load(path/"adapter.pt",weights_only=True))
            state=trainer.load_resume_state(path,ob,sb,accelerator,b.reference_contract)
            self.assertEqual(state["next_group_index"],1)
            self.assertEqual(step(b,ob,sb),expected_rewards)
            torch.testing.assert_close(a.model.weight,b.model.weight,rtol=0,atol=0)
            self.assertEqual(sa.state_dict(),sb.state_dict())
            with self.assertRaisesRegex(ValueError,"contract mismatch"):
                trainer.load_resume_state(path,ob,sb,accelerator,{"mode":"grpo"})
            (path/"rng_rank_0.pt").unlink()
            with self.assertRaisesRegex(ValueError,"complete checkpoint"):
                trainer.load_resume_state(path,ob,sb,accelerator,b.reference_contract)

    def test_explicit_base_best_is_not_replaced_with_last_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"epoch_5").mkdir()
            (Path(directory)/"best_checkpoint.json").write_text(json.dumps(selection_record(0,None,.4,.2)))
            self.assertIsNone(resolve_adapter(None,directory))


if __name__=="__main__":unittest.main()
