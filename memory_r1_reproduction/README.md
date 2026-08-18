# Memory-R1 Reproduction

This is one cleaned reproduction workspace with two independent training paths.
They share a repository and configuration layout, but one script never invokes
or passes runtime output to the other.

```text
memory_r1_reproduction/
  memfactory/                 MemFactory Memory-R1 implementation
  memory_r1/                  standalone Memory-R1 Python package
  scripts/                    every train and data-build entry point
  configs/memory_r1/          standalone Memory-R1 train configs
  data/memfactory/            MemFactory data
  data/memory_r1/             standalone Memory-R1 data
  examples/                   shared input examples
  evaluation/                 MemFactory evaluation utilities
  requirements/               dependency lists for each path
```

## Scripts

- `scripts/train_mem_grpo.py`: MemFactory GRPO trainer.
- `scripts/run_MemR1.sh`: MemFactory launch configuration.
- `scripts/convert_locomo_to_memfactory_memr1.py`: LoCoMo to MemFactory converter.
- `scripts/build_manager_training_data.py`: Algorithm 1 Manager data builder.
- `scripts/build_answer_training_data.py`: Algorithm 2 Answer data builder; requires a trained Manager checkpoint.
- `scripts/train_memory_r1_manager.py`: standalone manager trainer.
- `scripts/train_memory_r1_answer.py`: standalone answer trainer.

The source directories `memfactory_upstream` and `memory_r1_original` are not
modified. The copied workspace excludes unrelated MemFactory MemoryAgent, RMM,
LongContext, retriever, placeholder, and launch-script implementations.
