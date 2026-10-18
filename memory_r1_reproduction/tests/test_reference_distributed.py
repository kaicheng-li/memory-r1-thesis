"""Real four-process CPU/Gloo test of baseline-style gradient averaging."""
from datetime import timedelta
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_manager_reference as trainer
from memfactory.lora import initialize_collectives, shard_rows
from test_reference_baseline import Policy

REWARDS=[[.5,.5,.5,.5],[0.,1.,0.,1.],[.1,.2,.7,.8],[1.,.3,.4,.8]]


def perform(rank,mode,world=1):
    policy=Policy([])
    optimizer=torch.optim.SGD(policy.model.parameters(),lr=.1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    args=SimpleNamespace(advantage_mode=mode,advantage_scale=1.,_reference_score=.5,
        variance_epsilon=1e-12,beta=0.,grad_clip_norm=100.,max_prompt_tokens=32,clip_epsilon=.2)
    rows=[{"prompt":"p","completion_ids":torch.tensor([i+1]),"reward":v,"valid":True}
          for i,v in enumerate(REWARDS[rank])]
    trainer.update_policy(policy,optimizer,scheduler,rows,args,world)
    return float(policy.model.weight.detach())


def worker(rank,init_file,output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=Path(init_file).resolve().as_uri(),rank=rank,world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        initialize_collectives("cpu",4,enable_cpu_sync=True)
        values={mode:perform(rank,mode,4) for mode in ("grpo","fixed-zero","raw-reference")}
        Path(output,f"rank{rank}.json").write_text(json.dumps(values))
    finally:
        dist.destroy_process_group()


class DistributedReferenceTests(unittest.TestCase):
    def test_padding_contract(self):
        shards=[shard_rows(list(range(150)),r,4) for r in range(4)]
        self.assertTrue(all(len(s)==38 for s in shards))
        self.assertEqual(sum(map(len,shards))*4,608)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),"Gloo backend unavailable")
    def test_four_ranks_match_four_question_average_including_zero_signal_rank(self):
        # Some Windows Torch wheels expose Gloo but contain no usable transport.
        # Keep the actual four-process test enabled on supported hosts.
        try:
            dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        except RuntimeError as exc:
            if sys.platform == "win32" and "unsupported gloo device" in str(exc):
                self.skipTest(f"Installed Windows Torch cannot initialize Gloo: {exc}")
            raise
        expected={mode:sum(perform(r,mode) for r in range(4))/4
                  for mode in ("grpo","fixed-zero","raw-reference")}
        with tempfile.TemporaryDirectory() as directory:
            context=mp.get_context("spawn")
            processes=[context.Process(target=worker,args=(r,str(Path(directory)/"rendezvous"),directory)) for r in range(4)]
            for process in processes:process.start()
            deadline=time.monotonic()+180
            try:
                for process in processes:process.join(max(0,deadline-time.monotonic()))
                self.assertTrue(all(not p.is_alive() for p in processes),"Distributed test timed out")
                self.assertTrue(all(p.exitcode==0 for p in processes),[p.exitcode for p in processes])
                for rank in range(4):
                    actual=json.loads(Path(directory,f"rank{rank}.json").read_text())
                    for mode in expected:self.assertAlmostEqual(actual[mode],expected[mode],places=6)
            finally:
                for process in processes:
                    if process.is_alive():process.terminate()
                for process in processes:process.join(5)


if __name__=="__main__":unittest.main()
