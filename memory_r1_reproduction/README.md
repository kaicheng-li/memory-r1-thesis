# Memory-R1 Reproduction

This is the Memory-R1 reproduction workspace. Algorithm 5 is the single
training entry point; Algorithm 3 and Algorithm 4 are standalone inference
and data-construction scripts.

```text
memory_r1_reproduction/
  memfactory/                 prompts and memory-bank components
  scripts/                    Algorithm 3/4/5 and data-build entry points
  data/memfactory/            MemFactory data
  evaluation/                 MemFactory evaluation utilities
  requirements/               dependency lists
```

## Scripts

- `scripts/RLtrain.py`: Algorithm 5 Memory Manager PPO/GRPO trainer. Primary input is Algorithm 1 tuples (per-turn episodes); raw LoCoMo JSON is accepted as a full-dialogue fallback.
- `scripts/run_MemR1.sh`: Algorithm 5 launch script.
- `scripts/construct_memory_bank.py`: Algorithm 3 memory-bank construction.
- `scripts/generate_memory_augmented_answers.py`: Algorithm 4 answer generation.
- `scripts/build_manager_training_data.py`: Algorithm 1 Manager data builder.
- `scripts/build_answer_training_data.py`: Algorithm 2 Answer data builder; requires a trained Manager checkpoint.
- `scripts/train_answer_grpo.py`: GRPO training of the Answer Agent (paper Section 3.3) on Algorithm 2 tuples; the trained checkpoint is used as `--answer-model` by Algorithm 5.

`scripts/train_mem_grpo.py` remains the old MemFactory trainer and is not used
by the Memory-R1 Algorithm 5 entry point.
