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
- `scripts/construct_memory_bank.py`: constructs a full dialogue bank from fixed cached turn facts by default (`--fact-source cached`). Input must contain `turns[*].facts`, as in Stage-0 timelines. `--fact-source manager` explicitly selects the legacy policy-dependent extractor and is a separate experimental condition.
- `scripts/generate_memory_augmented_answers.py`: Algorithm 4 answer generation.
- `scripts/build_manager_training_data.py`: Algorithm 1 Manager data builder — creates the baseline 24-turn replay windows, extracts per-turn facts, and attaches each evidence-bearing QA to the last evidence turn. A tuple is retained only when all of that QA's evidence lies in the replay window.
- `scripts/build_answer_training_data.py`: consumes `--manager-tuples`, replays each cached-fact window with a frozen initial Manager, and retrieves top-30 memories per participant for its linked QA. This aligns Answer training contexts with Manager reward contexts; it is an experimental pipeline variant, not an assertion of exact Algorithm-2 reproduction.
- `scripts/build_sequential_from_stage0.py`: joins cached timelines and stable QA bindings, validates their identities, and constructs empty-bank sequential tuples. `--history-window 24` uses 24 preceding turns plus the current turn; `--history-window -1` retains the complete causal prefix.
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

These are turn-provenance proxies, not semantic support checks. ADD assigns
the current source turn even for unrelated text; UPDATE retains historical
sources without verifying the new text. Do not call this verified evidence
retention. Manager and Answer prompts exclude provenance fields.

`--terminal-potential observed` (default) preserves the branch's auxiliary
terminal coverage objective and can change the optimal QA policy.
`--terminal-potential zero` cancels discounted shaping over a complete
trajectory up to its initial-state constant. With full return-to-go, this
does not automatically improve local credit assignment; see the Chinese
review report for the derivation and limitations.

Manager training optionally accepts `--validation-timelines`; the run script
uses the Stage-0 validation timeline by default. It evaluates the initial
policy and each epoch on full-dialogue greedy QA and writes
`best_checkpoint.json`. Epoch 0 (`adapter: null`) may remain best. Selection
uses multiset token F1 under the local alphanumeric normalizer; the legacy
unique-overlap F1 reward is also reported separately. Neither is claimed to
be a verified official LoCoMo evaluator. Both trainers expose `--seed` and
write data hashes, arguments and package versions to `run_manifest.json`.

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
The Answer Agent is first warm-started from fixed-Manager window replay examples, then frozen
while manager RL learns to produce a bank it can use; it never receives raw
dialogue at training or inference.

The implementation is in `scripts/train_manager_grpo.py`, mainly in
`evidence_potential`, `apply_decisions`, `train_from_tuples`, and
`apply_policy_update`. The shaping weight is configurable with
`--evidence-weight` or the `EVIDENCE_WEIGHT` environment variable; the default
is `0.5`.

### Prepare the evidence-aware data

The sequential tuples attach QA to the last annotated evidence turn.
Adapt the existing Stage-0 cached timelines without consuming their single-step
old-bank snapshots:

```bash
python scripts/build_sequential_from_stage0.py \
  --timelines data/stage0/dialogue_timelines_train.json \
  --states data/stage0/manager_states_train.jsonl \
  --split train --history-window 24 \
  --output data/review/manager_tuples_train_window24.jsonl

python scripts/build_answer_training_data.py \
  --manager-tuples data/review/manager_tuples_train_window24.jsonl \
  --split train \
  --output data/answer_training_data_evidence.jsonl \
  --manager-model /path/to/base-model
```

Set `DATA_PATH=data/review/manager_tuples_train_window24.jsonl` for the run
script, whose historical default filename is not populated automatically.
Use `--history-window -1` and a different output filename for full prefixes.
`ANSWER_MANAGER_PATH` is a base model path; supply an adapter separately with
`ANSWER_MANAGER_ADAPTER_PATH`. Freeze and reuse the same Answer adapter for
the eta=0 and eta>0 comparison.

The raw dataset has no field for the real time at which a benchmark QA was
asked. `turn_index` in the prepared tuples is therefore an explicit training
endpoint derived from the latest annotated evidence turn; it is not a claim
about an original QA timestamp. The current local Stage-0 training split has
152 category-1--4 questions, of which 150 have usable evidence bindings.
The 24-predecessor window retains 122 of these; full prefixes retain all 150.
Generated manifests record omitted QA identities and data hashes. This is a
local split and cache audit, not a claim about all versions of LoCoMo.

`scripts/train_mem_grpo.py` remains the old MemFactory trainer and is not used
by the Memory-R1 Algorithm 5 entry point.
