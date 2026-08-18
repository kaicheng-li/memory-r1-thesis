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

- `scripts/train_manager_grpo.py`: Algorithm 5 Memory Manager PPO/GRPO trainer. Per tuple it starts from an empty bank and replays the dialogue window (Algorithm 1 output); raw LoCoMo JSON is accepted as a full-dialogue fallback.
- `scripts/run_MemR1.sh`: trains the Answer Agent (Section 3.3) on Algorithm 2 tuples, then launches the Algorithm 5 Manager training with that checkpoint as the frozen `--answer-model`.
- `scripts/construct_memory_bank.py`: Algorithm 3 memory-bank construction (deployment/eval side; training-data banks are built by the Algorithm 1 teacher).
- `scripts/generate_memory_augmented_answers.py`: Algorithm 4 answer generation.
- `scripts/build_manager_training_data.py`: Algorithm 1 Manager data builder — GPT-4o-mini builds a temporal bank per turn (50-turn window, no op labels); each tuple also carries the window turns for Algorithm 5.
- `scripts/build_answer_training_data.py`: Algorithm 2 Answer data builder — retrieves top-60 from the teacher-built temporal banks in the Algorithm 1 output; no manager replay.
- `scripts/train_answer_grpo.py`: GRPO training of the Answer Agent (paper Section 3.3) on Algorithm 2 tuples; its checkpoint is used as `--answer-model` by Algorithm 5.

`scripts/train_mem_grpo.py` remains the old MemFactory trainer and is not used
by the Memory-R1 Algorithm 5 entry point.
