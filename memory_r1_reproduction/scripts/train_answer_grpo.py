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
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_retriever import build_answer_input


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
                "memories": row.get("retrieved_memories", row.get("memory_bank", [])),
                "gold": str(row.get("answer", row.get("gold_answer", ""))),
            }
        )
    if not samples:
        raise ValueError(f"No answer-training tuples found in {path}")
    return samples


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_answer(text: str) -> str:
    return text.rsplit("Answer:", 1)[-1].strip()


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
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def score_group(
    model,
    tokenizer,
    prompt: str,
    completions: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level log probabilities over each completion (action) region.

    Returns (log_probs, action_mask) shaped [G, L'] where L' is the longest
    completion length; positions outside a row's action region are masked.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    rows = []
    for completion in completions:
        completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        rows.append((completion_ids, torch.cat([prompt_ids, completion_ids])))

    max_length = max(len(ids) for _, ids in rows)
    batch_input = torch.full((len(rows), max_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(batch_input)
    action_mask = torch.zeros_like(batch_input, dtype=torch.bool)
    for index, (completion_ids, ids) in enumerate(rows):
        offset = max_length - len(ids)
        batch_input[index, offset:] = ids
        attention[index, offset:] = 1
        action_mask[index, offset + len(prompt_ids):] = True

    outputs = model(input_ids=batch_input, attention_mask=attention, use_cache=False)
    logits = outputs.logits[:, :-1, :]  # position p predicts token p + 1
    labels = batch_input[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return log_probs, action_mask[:, 1:].to(log_probs.device)


def grpo_step(
    actor,
    reference,
    tokenizer,
    optimizer,
    prompt: str,
    completions: list[str],
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    """One GRPO update for a group of G answers to the same (q, M_ret)."""
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    loss_value = 0.0
    for _ in range(args.num_iterations):
        log_probs, action_mask = score_group(actor, tokenizer, prompt, completions, args.device)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
        per_token = -torch.minimum(ratio * advantages.view(-1, 1), clipped * advantages.view(-1, 1))
        if reference is not None:
            with torch.no_grad():
                reference_log_probs, _ = score_group(reference, tokenizer, prompt, completions, args.device)
            log_ratio = reference_log_probs - log_probs
            per_token = per_token + args.beta * (log_ratio.exp() - 1 - log_ratio)
        per_token = per_token * action_mask
        loss = (per_token.sum(dim=1) / action_mask.sum(dim=1).clamp(min=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.item())
    return loss_value


def train(args: argparse.Namespace) -> None:
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype).to(device)
    actor.train()

    reference = None
    if args.beta > 0:
        reference = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype).to(device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate)
    samples = read_rows(args.data_path)

    for epoch in range(args.epochs):
        progress = tqdm(samples, desc=f"answer-grpo-epoch-{epoch + 1}")
        for sample in progress:
            prompt = build_answer_input(sample["question"], {"Memory Bank": sample["memories"]})
            completions = generate_completions(
                actor, tokenizer, prompt, args.num_generations, args.max_new_tokens, args.temperature, device
            )
            rewards = torch.tensor(
                [exact_match_reward(extract_answer(completion), sample["gold"]) for completion in completions],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                old_log_probs, _ = score_group(actor, tokenizer, prompt, completions, device)
            old_log_probs = old_log_probs.detach()

            loss = grpo_step(actor, reference, tokenizer, optimizer, prompt, completions, rewards, old_log_probs, args)
            progress.set_postfix(loss=loss, reward=float(rewards.mean().item()))

        output_dir = Path(args.output_dir) / f"epoch_{epoch + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        actor.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"saved Answer Agent checkpoint to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO train the Memory-R1 Answer Agent (paper Section 3.3).")
    parser.add_argument("--data-path", required=True, help="Algorithm 2 tuples (build_answer_training_data.py output)")
    parser.add_argument("--model-path", required=True, help="Base model to initialize the Answer Agent")
    parser.add_argument("--output-dir", default="output/memory_r1_answer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--num-iterations", type=int, default=1, help="Inner GRPO update steps per rollout group")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
