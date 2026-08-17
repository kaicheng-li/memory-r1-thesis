from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_r1.prompts import build_answer_prompt
from memory_r1.rewards import answer_exact_match_reward
from memory_r1.rl import SequenceValueHead, grpo_loss, ppo_update, sequence_logprob
from memory_r1.schemas import MemoryEntry
from memory_r1.utils import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Memory-R1 Answer Agent with PPO or GRPO.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", default="outputs/answer_agent")
    parser.add_argument("--algorithm", choices=["ppo", "grpo"], default="grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    critic = None
    critic_optimizer = None
    if args.algorithm == "ppo":
        critic = SequenceValueHead(args.model_name_or_path, dtype=dtype).to(device)
        critic_optimizer = optim.AdamW(critic.parameters(), lr=args.learning_rate * 10)

    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.learning_rate)
    rows = load_jsonl(args.train_file)

    actor.train()
    for epoch in range(args.epochs):
        progress = tqdm(rows, desc=f"answer-{args.algorithm}-epoch-{epoch + 1}")
        for row in progress:
            prompt = build_answer_prompt(
                row["question"],
                [MemoryEntry(**memory) for memory in row["retrieved_memories"]],
            )

            if args.algorithm == "grpo":
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                with torch.no_grad():
                    outputs = actor.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        num_return_sequences=args.num_generations,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                completions = [
                    tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
                    for output in outputs
                ]
                rewards = torch.tensor(
                    [answer_exact_match_reward(text, row["answer"]) for text in completions],
                    dtype=torch.float32,
                    device=device,
                )
                prompts = [prompt] * len(completions)
                loss = grpo_loss(
                    actor=actor,
                    reference_model=reference_model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    completions=completions,
                    rewards=rewards,
                    beta=args.beta,
                    group_size=args.num_generations,
                    device=device,
                )
                actor_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                progress.set_postfix(loss=float(loss.item()), reward=float(rewards.mean().item()))
            else:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                with torch.no_grad():
                    outputs = actor.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
                reward = torch.tensor(
                    [answer_exact_match_reward(completion, row["answer"])],
                    dtype=torch.float32,
                    device=device,
                )
                old_log_probs = sequence_logprob(actor, tokenizer, [prompt], [completion], device=device).detach()
                stats = ppo_update(
                    actor=actor,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    completions=[completion],
                    rewards=reward,
                    old_log_probs=old_log_probs,
                    clip_epsilon=args.clip_epsilon,
                    value_coef=args.value_coef,
                    device=device,
                )
                progress.set_postfix(loss=stats.loss, reward=stats.reward)

    actor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
