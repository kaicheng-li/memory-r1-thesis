# Memory-R1 Recreation

This directory contains a standalone, paper-faithful recreation of the core `Memory-R1` pipeline described in:

`Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning`

Scope:

- standalone implementation under `C:\project\mythesis\memory_r1_original`
- two agents: `Memory Manager` and `Answer Agent`
- four memory operations: `ADD`, `UPDATE`, `DELETE`, `NONE`
- top-k memory retrieval with per-speaker partitioning
- outcome-driven rewards with exact match
- PPO and GRPO training entry points
- LoCoMo-style data builders

Important constraints:

- This is a reimplementation from the paper text, not an official release.
- The paper relies on proprietary data construction steps with `GPT-4o-mini`; this code exposes that step as a pluggable fact extractor and includes a local heuristic fallback.
- The paper uses VERL on multi-GPU H100 clusters. This recreation is written in plain PyTorch + Transformers so it can be inspected and adapted locally.

## Layout

- `memory_r1/`: library code
- `scripts/build_manager_data.py`: build Memory Manager tuples
- `scripts/build_answer_data.py`: build Answer Agent tuples
- `train_manager.py`: train the Memory Manager with PPO or GRPO
- `train_answer.py`: train the Answer Agent with PPO or GRPO
- `configs/`: example configs
- `examples/sample_dialogues.json`: minimal input example

## Directory Guide

This section explains what each directory and script does.

### Top level

- `README.md`
  - project overview
  - input format
  - training entry commands
  - architecture notes

- `requirements.txt`
  - minimal Python dependencies for this recreation

- `train_manager.py`
  - RL training entry for the `Memory Manager`
  - supports `ppo` and `grpo`
  - samples manager outputs, updates a temporary memory bank, then uses the frozen `Answer Agent` result as reward

- `train_answer.py`
  - RL training entry for the `Answer Agent`
  - supports `ppo` and `grpo`
  - trains the answering policy from retrieved memories and exact-match reward

### `configs/`

- `answer_train.example.json`
  - example hyperparameter config for training the `Answer Agent`

- `manager_train.example.json`
  - example hyperparameter config for training the `Memory Manager`

### `examples/`

- `sample_dialogues.json`
  - minimal dialogue example
  - used to test the data builders and show the expected dataset format

### `scripts/`

- `build_manager_data.py`
  - converts raw dialogue JSON into `Memory Manager` training tuples
  - output contains:
    - memory snapshot before current turn
    - current turn
    - linked question
    - gold answer

- `build_answer_data.py`
  - converts raw dialogue JSON into `Answer Agent` training tuples
  - output contains:
    - question
    - retrieved candidate memories
    - gold answer

### `memory_r1/`

This is the main library directory. The rest of the project calls into these modules.

- `__init__.py`
  - package export file
  - exposes the main public classes

- `schemas.py`
  - defines core data structures
  - includes:
    - dialogue turn
    - question-answer pair
    - memory entry
    - memory decision
    - manager sample
    - answer sample

- `utils.py`
  - common helpers
  - JSON and JSONL read/write
  - text normalization
  - exact-match reward helper
  - retrieval tokenization helper

- `prompts.py`
  - stores prompt templates
  - builds prompts for:
    - fact extraction
    - memory manager decisions
    - answer generation

- `parser.py`
  - parses model outputs back into structured Python objects
  - handles:
    - extracted facts
    - manager decision JSON
    - answer agent JSON

- `retriever.py`
  - retrieval layer over the memory bank
  - current implementation is lexical overlap scoring
  - also supports per-speaker retrieval to match the paper's `top 30 per participant` pattern

- `memory_bank.py`
  - in-memory storage for all memory entries
  - supports:
    - add
    - update
    - delete
    - apply a batch of manager decisions

- `agents.py`
  - model-facing agent wrappers
  - includes:
    - `HFTextGenerator`: generic Hugging Face generation wrapper
    - `FactExtractor`: turns a dialogue turn into durable facts
    - `MemoryManagerAgent`: predicts `ADD/UPDATE/DELETE/NONE`
    - `AnswerAgent`: selects useful memories and answers the question

- `datasets.py`
  - training data construction logic
  - builds:
    - manager training samples from turns plus memory snapshots
    - answer training samples from questions plus retrieved memories

- `rewards.py`
  - reward functions used by RL training
  - current main reward is exact match on the final answer

- `rl.py`
  - low-level RL utilities
  - sequence value head for PPO
  - sequence log-probability computation
  - PPO update logic
  - GRPO loss logic

- `pipeline.py`
  - end-to-end orchestration module
  - glues together:
    - fact extraction
    - retrieval
    - memory update
    - answer generation
  - this is the clearest place to understand the whole system flow

### `data/`

- `manager_train.jsonl`
  - generated training samples for `train_manager.py`
  - created by `scripts/build_manager_data.py`

- `answer_train.jsonl`
  - generated training samples for `train_answer.py`
  - created by `scripts/build_answer_data.py`

This directory is generated output, not source code.

## Expected dialogue format

Input JSON is a list of dialogues:

```json
[
  {
    "dialogue_id": "dlg-1",
    "participants": ["Andrew", "Audrey"],
    "turns": [
      {
        "speaker": "Andrew",
        "text": "I adopted a dog named Buddy.",
        "timestamp": "2023-01-01 20:30"
      }
    ],
    "questions": [
      {
        "question": "How many dogs did Andrew adopt?",
        "answer": "2",
        "turn_index": 1
      }
    ]
  }
]
```

`turn_index` points to the dialogue turn most closely associated with the QA pair.

## Quick start

1. Build manager tuples:

```powershell
python .\scripts\build_manager_data.py `
  --input .\examples\sample_dialogues.json `
  --output .\data\manager_train.jsonl
```

2. Build answer tuples:

```powershell
python .\scripts\build_answer_data.py `
  --input .\examples\sample_dialogues.json `
  --output .\data\answer_train.jsonl
```

3. Train the Answer Agent:

```powershell
python .\train_answer.py `
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct `
  --train_file .\data\answer_train.jsonl `
  --algorithm grpo
```

4. Train the Memory Manager with a frozen Answer Agent:

```powershell
python .\train_manager.py `
  --manager_model_name_or_path Qwen/Qwen2.5-3B-Instruct `
  --answer_model_name_or_path Qwen/Qwen2.5-3B-Instruct `
  --train_file .\data\manager_train.jsonl `
  --algorithm ppo
```

## Paper-faithful defaults

- Memory Manager data window: previous `50` turns
- Answer retrieval: top `30` memories per participant, total `60`
- training reward: exact match on final answer
- eval decoding: greedy
- train decoding: sampling with temperature `1.0`

## What to adapt first

- replace the heuristic fact extractor with your preferred extractor
- swap lexical retrieval for embedding retrieval
- move training loops to TRL or VERL if you need scale
- plug in the real LoCoMo / MSC / LongMemEval preprocessors
