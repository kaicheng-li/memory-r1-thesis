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

- `scripts/train_manager_grpo.py`: Algorithm 5 Memory Manager PPO/GRPO trainer. Per tuple it starts from an empty bank and replays the dialogue window using the teacher-extracted per-turn facts (LLMExtract, Algorithm 1 output).
- `scripts/run_MemR1.sh`: trains the Answer Agent (Section 3.3) on Algorithm 2 tuples, then launches the Algorithm 5 Manager training with that checkpoint as the frozen `--answer-model`.
- `scripts/construct_memory_bank.py`: Algorithm 3 memory-bank construction (deployment/eval side; LLMExtract runs with the manager model itself; training-data banks are built by the Algorithm 1 teacher).
- `scripts/generate_memory_augmented_answers.py`: Algorithm 4 answer generation.
- `scripts/build_manager_training_data.py`: Algorithm 1 Manager data builder — creates the baseline 24-turn replay windows, extracts per-turn facts, and attaches each evidence-bearing QA to the last evidence turn. A tuple is retained only when all of that QA's evidence lies in the replay window.
- `scripts/build_answer_training_data.py`: Algorithm 2 Answer data builder — runs a Manager checkpoint over the raw dialogues to construct the memory bank, then retrieves top-30 memories per participant for each QA.
- `scripts/train_answer_grpo.py`: GRPO training of the Answer Agent (paper Section 3.3) on Algorithm 2 tuples; its checkpoint is used as `--answer-model` by Algorithm 5.

## Evidence-Aware Branch

The current Git branch adds the final planned optimization on top of the
Memory-R1 baseline: raw dialogue evidence is used to construct evidence-
complete manager tuples and to calculate manager reward. It is never inserted
into the Answer Agent prompt. The frozen Answer Agent still receives the
question and retrieved memory only, so it cannot bypass the memory bank.

Each generated memory entry records `source_turn_ids`. For every linked QA,
the trainer computes a privileged potential from two quantities:

- memory evidence coverage: annotated evidence ids present in the memory bank
- retrieval evidence coverage: annotated evidence ids present in the retrieved memories, using the paper's top-30-per-participant retrieval

For manager action (t), the dense reward is the potential difference:

```text
r_dense(t) = evidence_weight * (gamma * Phi(t) - Phi(t - 1))
```

The final QA reward is evaluated on the memory state at the tuple endpoint,
which is the last evidence turn for each linked QA. It is added to the last
manager action in that tuple. Return-to-go then propagates it to earlier
manager actions:

```text
G(t) = r(t) + gamma * G(t + 1)
```

The return-to-go values are normalized across sampled trajectories at the same
replayed turn position. This is the evidence-aware extension of the R1
manager update; it does not claim to remove the rollout-state divergence that
Memory-R2 addresses with local rerolls. Training and deployment use the same
memory transition and retrieval implementation in `memfactory/memory_runtime.py`.
The Answer Agent is first warm-started from teacher-bank examples, then frozen
while manager RL learns to produce a bank it can use; it never receives raw
dialogue at training or inference.

The implementation is in `scripts/train_manager_grpo.py`, mainly in
`evidence_potential`, `apply_decisions`, `train_from_tuples`, and
`apply_policy_update`. The shaping weight is configurable with
`--evidence-weight` or the `EVIDENCE_WEIGHT` environment variable; the default
is `0.5`.

### Prepare the evidence-aware data

This branch requires rebuilt datasets because the baseline tuples attach QA to
the first evidence turn. The evidence-aware tuples attach QA to the last
evidence turn and use the baseline 24-turn replay window. Keep the baseline files intact
and build separate files:

```bash
python scripts/build_manager_training_data.py \
  --input data/locomo10.json \
  --output data/manager_training_data_evidence.jsonl \
  --cache data/nvidia_teacher_cache.json

python scripts/build_answer_training_data.py \
  --input data/locomo10.json \
  --output data/answer_training_data_evidence.jsonl \
  --manager-model /path/to/memory-manager-checkpoint
```

`scripts/run_MemR1.sh` defaults to these two evidence-aware files. Its
`ANSWER_MANAGER_PATH` defaults to the base Manager checkpoint as an explicit
bootstrap; set it to a trained Manager checkpoint when reproducing Algorithm 2
literally.

The raw dataset has no field for the real time at which a benchmark QA was
asked. `turn_index` in the prepared tuples is therefore an explicit training
endpoint derived from the latest annotated evidence turn; it is not a claim
about an original QA timestamp. Four raw QA items have no evidence and five
raw QA items contain an evidence id absent from their conversation; those
items are not used for this evidence-aware manager reward. The raw data also
contains compound evidence strings separated by semicolons or spaces, which
the builder splits.
With the baseline 24-turn window, only evidence-complete QA windows are
linked, so long-range evidence outside a window is excluded rather than filled
with future information. In the checked `locomo10.json`, this leaves 1670 of
1977 valid evidence-bearing QA items for the 24-turn manager tuples.

`scripts/train_mem_grpo.py` remains the old MemFactory trainer and is not used
by the Memory-R1 Algorithm 5 entry point.
