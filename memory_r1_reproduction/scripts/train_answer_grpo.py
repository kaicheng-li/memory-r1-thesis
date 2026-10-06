"""GRPO training for the Answer Agent (Memory-R1 paper, Section 3.3).

The Answer Agent is a policy y ~ pi_ans(· | q, M_ret) that maps a question and
its retrieved memories to an answer. Following the paper:

1. For each (q, M_ret) sample G candidate answers with temperature 1.0.
2. Score every completion with the exact-match reward R_answer = EM(y_pred, y_gold).
3. Normalize rewards into group-relative advantages.
4. Update with the clipped GRPO objective plus a KL penalty against the
   reference model (beta * (exp(ref_lp - act_lp) - 1 - (ref_lp - act_lp))).

The Memory Manager is not touched here: it only appears implicitly through the
pre-computed training tuples (Algorithm 2 output of
scripts/build_answer_training_data.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_retriever import build_answer_input
from memfactory.chat import tokenize_chat_prompt
from memfactory.lora import (
    average_gradients,
    initialize_collectives,
    load_lora_model,
    shard_rows,
    trainable_parameters,
)


def read_rows(path: str) -> list[dict[str, Any]]:
    """Read JSONL or JSON answer-training tuples (Algorithm 2 output)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else [raw]
    samples = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = row.get("question", row.get("query", ""))
        if not question:
            continue
        samples.append(
            {
                "question": str(question),
                "dialogue_id": str(row.get("dialogue_id", "dialogue")),
                "question_id": str(row.get("question_id", "")),
                "memories": row.get("retrieved_memories", row.get("memory_bank", [])),
                "participants": row.get("participants", []),
                "gold": str(row.get("answer", row.get("gold_answer", ""))),
            }
        )
    if not samples:
        raise ValueError(f"No answer-training tuples found in {path}")
    return samples


def select_answer_split(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], int, int]:
    """Filter invalid tuples and optionally apply the 1:1:8 split."""
    valid = [row for row in rows if row["memories"] and row["gold"].strip()]
    if split == "all":
        return valid, len(rows), len(valid)
    train_end = int(len(valid) * 0.1 + 0.5)
    validation_end = int(len(valid) * 0.2 + 0.5)
    boundaries = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(valid)),
    }
    start, end = boundaries[split]
    return valid[start:end], len(rows), len(valid)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_answer(text: str) -> str:
    tagged = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    marked = re.search(r"\*\*Answer:\*\*\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if marked:
        return marked.group(1).strip()
    answer = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return answer.group(1).strip() if answer else text.strip()


def exact_match_reward(prediction: str, gold: str) -> float:
    """R_answer = EM(y_pred, y_gold), the paper's reward for the Answer Agent."""
    prediction_tokens = normalize_text(prediction)
    gold_tokens = normalize_text(gold)
    return 1.0 if prediction_tokens and prediction_tokens == gold_tokens else 0.0


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    max_prompt_tokens: int,
) -> list[tuple[str, torch.Tensor]]:
    """Sample G answers; returns (decoded text, raw sampled token ids) pairs.

    The token ids are kept so that score_group evaluates the exact action the
    policy sampled, not a re-tokenized reconstruction of the decoded text.
    """
    inputs = tokenize_chat_prompt(tokenizer, prompt, max_prompt_tokens).to(device)
    was_training = model.training
    was_checkpointing = model.is_gradient_checkpointing
    model.eval()
    if was_checkpointing:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    model.config.use_cache = False
    if was_checkpointing:
        model.gradient_checkpointing_enable()
    model.train(was_training)
    prompt_length = inputs["input_ids"].shape[1]
    completions = []
    for output in outputs:
        ids = output[prompt_length:]
        eos_positions = (ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if len(eos_positions):
            ids = ids[: int(eos_positions[0]) + 1]
        ids = ids.clone()
        completions.append((tokenizer.decode(ids, skip_special_tokens=True).strip(), ids))
    return completions


def score_group(
    model,
    tokenizer,
    prompt: str,
    completion_ids: list[torch.Tensor],
    device: str,
    micro_batch_size: int,
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level log probabilities over the sampled completion token ids.

    Returns (log_probs, action_mask) shaped [G, L'] where L' is the longest
    completion length; positions outside a row's action region are masked.
    """
    prompt_ids = tokenize_chat_prompt(tokenizer, prompt, max_prompt_tokens)["input_ids"][0]
    scores = []
    for start in range(0, len(completion_ids), micro_batch_size):
        scores.extend(score_batch(model, prompt_ids, completion_ids[start : start + micro_batch_size], device, tokenizer.pad_token_id))
    max_length = max(len(score) for score in scores)
    log_probs = torch.zeros((len(scores), max_length), device=device)
    action_mask = torch.zeros((len(scores), max_length), dtype=torch.bool, device=device)
    for index, score in enumerate(scores):
        log_probs[index, : len(score)] = score
        action_mask[index, : len(score)] = True
    return log_probs, action_mask


def score_batch(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: list[torch.Tensor],
    device: str,
    pad_token_id: int,
) -> list[torch.Tensor]:
    """Score a small completion batch without allocating logits for all G samples."""
    prompt_ids = prompt_ids.to(device)
    sequences = [torch.cat([prompt_ids, ids.to(device)]) for ids in completion_ids]
    max_length = max(len(sequence) for sequence in sequences)
    batch_input = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(batch_input)
    offsets = []
    for index, sequence in enumerate(sequences):
        offset = max_length - len(sequence)
        offsets.append(offset)
        batch_input[index, offset:] = sequence
        attention[index, offset:] = 1

    causal_lm = model.get_base_model() if hasattr(model, "peft_config") else model
    hidden_states = causal_lm.model(
        input_ids=batch_input,
        attention_mask=attention,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    action_states = []
    action_labels = []
    lengths = []
    for index, (ids, offset) in enumerate(zip(completion_ids, offsets)):
        start = offset + prompt_ids.numel() - 1
        action_states.append(hidden_states[index, start : start + len(ids)])
        action_labels.append(batch_input[index, start + 1 : start + 1 + len(ids)])
        lengths.append(len(ids))

    logits = causal_lm.lm_head(torch.cat(action_states))
    labels = torch.cat(action_labels)
    log_probs = logits.gather(1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return list(log_probs.split(lengths))


def grpo_step(
    actor,
    tokenizer,
    optimizer,
    prompt: str,
    completion_ids: list[torch.Tensor],
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    args: argparse.Namespace,
    num_processes: int,
) -> float:
    """One GRPO update for a group of G answers to the same (q, M_ret)."""
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    loss_value = 0.0
    reference_log_probs = None
    if args.beta > 0:
        was_training = actor.training
        actor.eval()
        with torch.no_grad():
            with actor.disable_adapter():
                reference_log_probs = score_group(
                    actor, tokenizer, prompt, completion_ids, args.device,
                    args.score_micro_batch, args.max_prompt_tokens,
                )[0]
        actor.train(was_training)

    optimizer.zero_grad()
    prompt_ids = tokenize_chat_prompt(tokenizer, prompt, args.max_prompt_tokens)["input_ids"][0]
    for start in range(0, len(completion_ids), args.score_micro_batch):
        ids_batch = completion_ids[start : start + args.score_micro_batch]
        log_probs_batch = score_batch(
            actor, prompt_ids, ids_batch, args.device, tokenizer.pad_token_id
        )
        chunk_losses = []
        for offset, log_probs in enumerate(log_probs_batch):
            index = start + offset
            old = old_log_probs[index, : len(log_probs)]
            ratio = torch.exp(log_probs - old)
            clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
            per_token = -torch.minimum(
                ratio * advantages[index], clipped * advantages[index]
            )
            if reference_log_probs is not None:
                log_ratio = reference_log_probs[index, : len(ids_batch[offset])] - log_probs
                per_token = per_token + args.beta * (log_ratio.exp() - 1 - log_ratio)
            chunk_losses.append(per_token.mean())
        chunk_loss = torch.stack(chunk_losses).sum() / len(completion_ids)
        chunk_loss.backward()
        loss_value += float(chunk_loss.detach().item())
    average_gradients(actor, num_processes)
    torch.nn.utils.clip_grad_norm_(trainable_parameters(actor), 1.0)
    optimizer.step()
    return loss_value


def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    device = str(accelerator.device)
    args.device = device
    set_seed(args.seed)
    initialize_collectives(device, accelerator.num_processes)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    actor = load_lora_model(args.model_path, device, trainable=True)
    optimizer = torch.optim.AdamW(trainable_parameters(actor), lr=args.learning_rate)
    all_samples = read_rows(args.data_path)
    samples_for_split, raw_count, valid_count = select_answer_split(all_samples, args.split)
    if accelerator.is_main_process:
        split_note = "after filtering" if args.split == "all" else "after 1:1:8 split"
        print(
            f"Answer data: {raw_count} raw, {valid_count} valid after filtering, "
            f"{len(samples_for_split)} {args.split} QA pairs {split_note}"
        )
    samples = shard_rows(samples_for_split, accelerator.process_index, accelerator.num_processes)

    for epoch in range(args.epochs):
        progress = tqdm(
            samples,
            desc=f"answer-grpo-epoch-{epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )
        first_group_logged = False
        for sample in progress:
            memories_by_speaker = {
                participant: [
                    memory for memory in sample["memories"]
                    if memory["speaker"] == participant
                ]
                for participant in sample["participants"]
            }
            prompt = build_answer_input(sample["question"], memories_by_speaker)
            rollout_start = time.perf_counter()
            completions = generate_completions(
                actor, tokenizer, prompt, args.num_generations, args.max_new_tokens,
                args.temperature, device, args.max_prompt_tokens,
            )
            rollout_seconds = time.perf_counter() - rollout_start
            decoded = [text for text, _ in completions]
            completion_ids = [ids for _, ids in completions]
            rewards = torch.tensor(
                [exact_match_reward(extract_answer(text), sample["gold"]) for text in decoded],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                old_log_probs, _ = score_group(
                    actor,
                    tokenizer,
                    prompt,
                    completion_ids,
                    device,
                    args.score_micro_batch,
                    args.max_prompt_tokens,
                )
            if accelerator.is_main_process and not first_group_logged:
                print(
                    "Answer first group: "
                    + json.dumps(
                        {
                            "gold": sample["gold"],
                            "rewards": rewards.tolist(),
                            "completions": decoded,
                            "extracted": [extract_answer(text) for text in decoded],
                        },
                        ensure_ascii=False,
                    )
                )
                first_group_logged = True
            update_start = time.perf_counter()
            loss = grpo_step(
                actor,
                tokenizer,
                optimizer,
                prompt,
                completion_ids,
                rewards,
                old_log_probs.detach(),
                args,
                accelerator.num_processes,
            )
            update_seconds = time.perf_counter() - update_start
            progress.set_postfix(
                loss=loss,
                reward=float(rewards.mean().item()),
                hits=int(rewards.sum().item()),
                tokens=sum(len(ids) for ids in completion_ids),
                rollout_s=round(rollout_seconds, 1),
                update_s=round(update_seconds, 1),
            )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            output_dir = Path(args.output_dir) / f"epoch_{epoch + 1}"
            output_dir.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"saved Answer Agent checkpoint to {output_dir}")
        accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO train the Memory-R1 Answer Agent (paper Section 3.3).")
    parser.add_argument("--data-path", required=True, help="Algorithm 2 tuples (build_answer_training_data.py output)")
    parser.add_argument("--model-path", required=True, help="Base model to initialize the Answer Agent")
    parser.add_argument("--output-dir", default="output/memory_r1_answer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--score-micro-batch", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="train")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
