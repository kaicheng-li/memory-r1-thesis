# Memory-R1 Reproduction

This is the Memory-R1 reproduction workspace. Manager training follows the
paper's Section 3.1 single-state semantics: G actions are sampled from a shared
oracle-history snapshot. Full dialogue replay is used only for deployment and
roll-forward validation.

```text
memory_r1_reproduction/
  memfactory/                 prompts and memory-bank components
  scripts/                    Algorithm 3/4/5 and data-build entry points
  data/memfactory/            MemFactory data
  evaluation/                 MemFactory evaluation utilities
  requirements/               dependency lists
```

## Scripts

- `scripts/rebuild_manager_states.py`: deterministically rebuilds provenance-preserving single-step states and cached-fact dialogue timelines. QA IDs are frozen on the raw array before category-5 filtering.
- `scripts/precheck_counterfactual.py`: paired NOOP/oracle Stage-0.5 gate. It can rescore saved predictions on CPU.
- `scripts/train_manager_grpo.py`: shared-snapshot single-step Manager GRPO with continuous token-F1 reward, strict action validation, complete resume checkpoints, and single-step plus roll-forward validation.
- `scripts/run_MemR1.sh`: three-stage pipeline: selected Manager, Answer-data construction from cached facts, then Answer training. It refuses to train without a passed Stage-0.5 gate.
- `scripts/construct_memory_bank.py`: deployment-side memory-bank construction using the same strict Manager action contract.
- `scripts/generate_memory_augmented_answers.py`: Algorithm 4 answer generation; retrieves top-30 memories per participant from the Algorithm 3 memory bank.
- `scripts/build_manager_training_data.py`: legacy teacher-fact cache source. Its cached per-turn facts are consumed by the Stage-0 rebuilder, not by the trainer.
- `scripts/build_answer_training_data.py`: rolls the selected Manager over Stage-0 cached-fact timelines and retrieves top-30 memories per participant.
- `scripts/train_answer_grpo.py`: Answer-Agent GRPO with continuous token-F1 reward, zero-variance skipping, and greedy validation checkpoint selection.
- `evaluation/run_checkpoint_audit.sh`: compares base, bootstrap, Answer, and Manager checkpoints. It reports the paper's LoCoMo F1/BLEU-1 metrics by question type and overall, plus EM and Manager validity diagnostics. Set `RUN_LLM_JUDGE=1` to add the paper's CORRECT/WRONG LLM-as-a-Judge metric through an OpenAI-compatible endpoint; existing JSONL results are rescored without repeating model generation.

The audit defaults to a full deterministic training-split audit on every QA,
with one greedy generation per checkpoint and round-robin execution across
`AUDIT_GPUS` (default `0,1,2,3`). The output directory includes the generation
count and temperature so stochastic and greedy audits cannot be mixed. A
Manager checkpoint audit includes epochs 4 and 5 by default; set
`MANAGER_EPOCHS=1,2,3,4,5` when all saved Manager epochs are needed. The
checkpoint paths are resolved below `MANAGER_DIR` (for example,
`MANAGER_DIR/epoch_4`). A
paper-comparable LoCoMo result requires `AUDIT_SPLIT=test`, full samples
(`ANSWER_SAMPLE_COUNT=0 MANAGER_SAMPLE_COUNT=0`), and Answer/Manager data built
from the official test dialogues. The script rejects Answer data from a
different split instead of silently reporting it as test performance. The
reported scorer is `memory_r1_locomo_v1`: article-stripped token F1, clipped
BLEU-1 with brevity penalty, and the paper's binary CORRECT/WRONG judge prompt.
Use one greedy generation for paper-style evaluation. The summary prints
`J_n` so failed or missing judge calls cannot be mistaken for evaluated zeros.
The judge model/configuration is stored with every label; changing it forces a
fresh J pass so scores from different judges are not silently mixed.
It also reports p50/p95 generation and retrieval latency for the Manager,
memory search, and Answer Agent. MSC and LongMemEval are not bundled in this
repository; those zero-shot tables require their external evaluation datasets
and are not synthesized from LoCoMo audit rows.

`scripts/train_mem_grpo.py` remains the old MemFactory trainer and is not used
by the single-step Memory-R1 pipeline.
