from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_r1.agents import AnswerAgent
from memory_r1.memory_bank import MemoryBank
from memory_r1.parser import parse_manager_output
from memory_r1.prompts import build_manager_prompt
from memory_r1.retriever import LexicalMemoryRetriever
from memory_r1.rewards import safe_reward
from memory_r1.rl import SequenceValueHead, grpo_loss, ppo_update, sequence_logprob
from memory_r1.schemas import DialogueTurn, MemoryEntry
from memory_r1.utils import load_jsonl


def _entry(memory_dict):
    return MemoryEntry(**memory_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Memory-R1 Memory Manager with PPO or GRPO.")
    parser.add_argument("--manager_model_name_or_path", required=True)
    parser.add_argument("--answer_model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", default="outputs/memory_manager")
    parser.add_argument("--algorithm", choices=["ppo", "grpo"], default="ppo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=1e-5)
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

    tokenizer = AutoTokenizer.from_pretrained(args.manager_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor = AutoModelForCausalLM.from_pretrained(
        args.manager_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.manager_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    critic = None
    critic_optimizer = None
    if args.algorithm == "ppo":
        critic = SequenceValueHead(args.manager_model_name_or_path, dtype=dtype).to(device)
        critic_optimizer = optim.AdamW(critic.parameters(), lr=args.critic_learning_rate)

    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_name_or_path, trust_remote_code=True)
    if answer_tokenizer.pad_token is None:
        answer_tokenizer.pad_token = answer_tokenizer.eos_token
    answer_model = AutoModelForCausalLM.from_pretrained(
        args.answer_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    answer_agent = AnswerAgent(generator=type("Generator", (), {
        "tokenizer": answer_tokenizer,
        "model": answer_model,
        "device": device,
        "generate": lambda self, prompt, max_new_tokens=256, temperature=0.0, do_sample=None, num_return_sequences=1: _generate_texts(
            self.model, self.tokenizer, self.device, prompt, max_new_tokens, temperature, do_sample, num_return_sequences
        ),
    })())

    retriever = LexicalMemoryRetriever()
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.learning_rate)
    rows = load_jsonl(args.train_file)

    actor.train()
    for epoch in range(args.epochs):
        progress = tqdm(rows, desc=f"manager-{args.algorithm}-epoch-{epoch + 1}")
        for row in progress:
            turn = DialogueTurn(**row["current_turn"])
            memories = [_entry(memory) for memory in row["memory_snapshot"]]
            bank = MemoryBank(memories)
            facts = [f"{turn.speaker}: {turn.text}"]
            retrieved = retriever.retrieve(turn.text, memories, top_k=5)
            retrieved_entries = [item.entry for item in retrieved]
            prompt = build_manager_prompt(turn, facts, retrieved_entries)

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
                rewards = []
                for completion in completions:
                    candidate_bank = bank.clone()
                    try:
                        decisions = parse_manager_output(completion)
                        candidate_bank.apply(decisions, default_speaker=turn.speaker, timestamp=turn.timestamp)
                        answer_retrieved = retriever.retrieve_per_speaker(
                            row["question"], candidate_bank.entries(), row["participants"], per_speaker_top_k=30
                        )
                        trace = answer_agent.answer(row["question"], answer_retrieved, temperature=0.0)
                        rewards.append(safe_reward(trace.answer, row["answer"], parse_ok=True))
                    except Exception:
                        rewards.append(safe_reward("", row["answer"], parse_ok=False))
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                prompts = [prompt] * len(completions)
                loss = grpo_loss(
                    actor=actor,
                    reference_model=reference_model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    completions=completions,
                    rewards=rewards_tensor,
                    beta=args.beta,
                    group_size=args.num_generations,
                    device=device,
                )
                actor_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                progress.set_postfix(loss=float(loss.item()), reward=float(rewards_tensor.mean().item()))
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
                parse_ok = True
                try:
                    decisions = parse_manager_output(completion)
                    bank.apply(decisions, default_speaker=turn.speaker, timestamp=turn.timestamp)
                    answer_retrieved = retriever.retrieve_per_speaker(
                        row["question"], bank.entries(), row["participants"], per_speaker_top_k=30
                    )
                    trace = answer_agent.answer(row["question"], answer_retrieved, temperature=0.0)
                    reward_value = safe_reward(trace.answer, row["answer"], parse_ok=True)
                except Exception:
                    parse_ok = False
                    reward_value = safe_reward("", row["answer"], parse_ok=False)
                reward = torch.tensor([reward_value], dtype=torch.float32, device=device)
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
                progress.set_postfix(loss=stats.loss, reward=stats.reward, parse_ok=parse_ok)

    actor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def _generate_texts(model, tokenizer, device, prompt, max_new_tokens, temperature, do_sample, num_return_sequences):
    do_sample = do_sample if do_sample is not None else temperature > 0
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip() for output in outputs]


if __name__ == "__main__":
    main()
