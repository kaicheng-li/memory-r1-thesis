from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class SequenceValueHead(nn.Module):
    def __init__(self, model_name_or_path: str, dtype: torch.dtype):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        hidden = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]
        values = self.value_head(last_hidden).squeeze(-1)
        last_index = attention_mask.long().sum(dim=1) - 1
        return values[torch.arange(values.shape[0], device=values.device), last_index]


@dataclass
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    reward: float


def sequence_logprob(model, tokenizer, prompts: List[str], completions: List[str], device: str) -> torch.Tensor:
    merged = [prompt + completion for prompt, completion in zip(prompts, completions)]
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    merged_ids = tokenizer(merged, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = merged_ids["input_ids"].clone()

    for i in range(labels.shape[0]):
        prompt_len = int(prompt_ids["attention_mask"][i].sum().item())
        labels[i, :prompt_len] = -100

    outputs = model(**merged_ids)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = token_log_probs.gather(2, shift_labels.masked_fill(shift_labels < 0, 0).unsqueeze(-1)).squeeze(-1)
    gathered = gathered.masked_fill(shift_labels < 0, 0.0)
    return gathered.sum(dim=1)


def ppo_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    tokenizer,
    prompts: List[str],
    completions: List[str],
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_epsilon: float,
    value_coef: float,
    device: str,
) -> PPOStats:
    new_log_probs = sequence_logprob(actor, tokenizer, prompts, completions, device=device)
    inputs = tokenizer(
        [p + c for p, c in zip(prompts, completions)],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    values = critic(inputs["input_ids"], inputs["attention_mask"])
    advantages = rewards - values.detach()

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, rewards)
    total_loss = policy_loss + value_coef * value_loss

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    total_loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()

    return PPOStats(
        loss=float(total_loss.item()),
        policy_loss=float(policy_loss.item()),
        value_loss=float(value_loss.item()),
        reward=float(rewards.mean().item()),
    )


def grpo_loss(
    actor,
    reference_model,
    tokenizer,
    prompts: List[str],
    completions: List[str],
    rewards: torch.Tensor,
    beta: float,
    group_size: int,
    device: str,
) -> torch.Tensor:
    actor_log_probs = sequence_logprob(actor, tokenizer, prompts, completions, device=device)
    with torch.no_grad():
        ref_log_probs = sequence_logprob(reference_model, tokenizer, prompts, completions, device=device)

    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = ((grouped - mean) / (std + 1e-4)).view(-1)

    kl = ref_log_probs - actor_log_probs
    return -(actor_log_probs * advantages - beta * kl).mean()
