"""Single-step shared-snapshot GRPO training for the Memory Manager.

Each Stage-0 row is one QA-conditioned state: an old bank containing only
turns before the target, current-turn facts, a question, and its gold answer.
The G samples share that state and each receives an independent bank copy.
The frozen Answer model decodes greedily and continuous token F1 is reward.
Dialogue replay exists only in roll-forward validation, never in training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.manager_validation import (
    evaluate_roll_forward,
    evaluate_single_step,
    read_json_or_jsonl,
)
from evaluation.metrics import exact_match, f1_score
from memfactory.chat import tokenize_chat_prompt
from memfactory.progress import append_progress
from memfactory.lora import (
    average_gradients,
    initialize_collectives,
    load_lora_model,
    shard_rows,
    sum_cpu_scalars,
    trainable_parameters,
)
from memfactory.manager_env import validate_and_apply
from memfactory.modules.memory_updater import build_manager_input
from scripts.construct_memory_bank import parse_manager_output, top_k, top_k_per_speaker


def extract_answer(text: str) -> str:
    tagged = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    marked = re.search(r"\*\*Answer:\*\*\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if marked:
        return marked.group(1).strip()
    answer = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return answer.group(1).strip() if answer else text.strip()


class TextModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        *,
        trainable: bool,
        adapter_path: str | None = None,
    ) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.model = load_lora_model(model_path, device, adapter_path, trainable)
        self.model.train(trainable)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        max_prompt_tokens: int,
    ) -> tuple[str, torch.Tensor]:
        inputs = tokenize_chat_prompt(
            self.tokenizer, prompt, max_prompt_tokens
        ).to(self.device)
        was_training = self.model.training
        was_checkpointing = self.model.is_gradient_checkpointing
        self.model.eval()
        if was_checkpointing:
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        generation: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            # Neutralize checkpoint defaults: rollout must match raw policy logits.
            "top_k": 0,
            "top_p": 1.0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation["temperature"] = temperature
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation)
        self.model.config.use_cache = False
        if was_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.train(was_training)
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_length:]
        eos = (completion_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=False)
        if len(eos):
            completion_ids = completion_ids[: int(eos[0]) + 1]
        completion_ids = completion_ids.clone()
        return (
            self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
            completion_ids,
        )

    def log_probability(
        self, prompt: str, completion_ids: torch.Tensor, max_prompt_tokens: int
    ) -> torch.Tensor:
        if completion_ids.numel() == 0:
            return torch.zeros((0,), device=self.device)
        prompt_ids = tokenize_chat_prompt(
            self.tokenizer, prompt, max_prompt_tokens
        )["input_ids"][0]
        input_ids = torch.cat(
            [prompt_ids.to(self.device), completion_ids.to(self.device)]
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        causal_lm = (
            self.model.get_base_model() if hasattr(self.model, "peft_config")
            else self.model
        )
        hidden = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        start = max(prompt_ids.numel() - 1, 0)
        action_states = hidden[:, start : start + completion_ids.numel(), :]
        labels = input_ids[:, start + 1 : start + 1 + completion_ids.numel()]
        logits = causal_lm.lm_head(action_states)
        selected = logits.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return (selected - torch.logsumexp(logits, dim=-1)).squeeze(0)


def read_training_states(path: str) -> list[dict[str, Any]]:
    rows = read_json_or_jsonl(path)
    required = {
        "dialogue_id", "question_id", "question", "answer", "participants",
        "target_turn", "target_speaker", "target_timestamp", "current_facts",
        "old_bank",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Stage-0 row {index} is missing fields: {missing}")
        if not row["current_facts"]:
            raise ValueError(
                f"Training row {row['dialogue_id']}:{row['question_id']} has no current facts"
            )
    if not rows:
        raise ValueError(f"No Stage-0 states found in {path}")
    return rows


def answer_from_bank(
    answer: TextModel,
    question: str,
    bank: list[dict[str, Any]],
    participants: list[str],
    args: argparse.Namespace,
) -> str:
    from memfactory.modules.memory_retriever import build_answer_input

    retrieved = top_k_per_speaker(
        question, bank, participants, args.answer_top_k_per_speaker
    )
    memories = {
        participant: [item for item in retrieved if item["speaker"] == participant]
        for participant in participants
    }
    prompt = build_answer_input(question, memories)
    raw, _ = answer.generate(
        prompt, args.answer_max_new_tokens, 0.0, args.max_prompt_tokens
    )
    return extract_answer(raw)


def rollout_group(
    manager: TextModel,
    answer: TextModel,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Sample G independent one-step actions from one shared state."""
    facts = state["current_facts"]
    query = " ".join(str(fact.get("text", "")) for fact in facts)
    visible = top_k(query, state["old_bank"], args.manager_top_k)
    prompt = build_manager_input(visible, facts)
    rollouts = []
    for _ in range(args.num_generations):
        raw, completion_ids = manager.generate(
            prompt, args.max_new_tokens, args.temperature, args.max_prompt_tokens
        )
        valid = False
        error = None
        actions = 0
        candidate_bank = list(state["old_bank"])
        try:
            decisions = parse_manager_output(raw)
            result = validate_and_apply(
                state["old_bank"], visible, decisions,
                dialogue_id=str(state["dialogue_id"]),
                source_turn=int(state["target_turn"]),
                speaker=str(state["target_speaker"]),
                timestamp=str(state["target_timestamp"]),
            )
            valid = result.valid
            error = result.error
            actions = result.applied_actions
            candidate_bank = result.memory_bank
        except (ValueError, json.JSONDecodeError) as exc:
            error = f"parse: {exc}"

        prediction = ""
        reward = 0.0
        strict_em = 0.0
        if valid:
            prediction = answer_from_bank(
                answer, state["question"], candidate_bank, state["participants"], args
            )
            reward = f1_score(prediction, state["answer"])
            strict_em = exact_match(prediction, state["answer"])
        rollouts.append({
            "prompt": prompt,
            "completion_ids": completion_ids,
            "valid": valid,
            "error": error,
            "actions": actions,
            "reward": reward,
            "em": strict_em,
        })
    return rollouts


def update_policy(
    manager: TextModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rollouts: list[dict[str, Any]],
    args: argparse.Namespace,
    num_processes: int,
) -> dict[str, Any]:
    rewards = torch.tensor(
        [item["reward"] for item in rollouts],
        dtype=torch.float32,
        device=manager.device,
    )
    reward_std = float(rewards.std(unbiased=False).item())
    local_signal = reward_std > args.variance_epsilon
    advantages = (
        (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        if local_signal else torch.zeros_like(rewards)
    )
    optimizer.zero_grad()
    loss_sum = 0.0
    trained_actions = 0
    if local_signal:
        available = sum(item["completion_ids"].numel() > 0 for item in rollouts)
        for index, item in enumerate(rollouts):
            completion_ids = item["completion_ids"]
            if completion_ids.numel() == 0:
                continue
            log_probs = manager.log_probability(
                item["prompt"], completion_ids, args.max_prompt_tokens
            )
            ratio = torch.exp(log_probs - log_probs.detach())
            clipped = torch.clamp(
                ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
            )
            per_token = -torch.minimum(
                ratio * advantages[index], clipped * advantages[index]
            )
            if args.beta > 0:
                was_training = manager.model.training
                manager.model.eval()
                with torch.no_grad(), manager.model.disable_adapter():
                    reference = manager.log_probability(
                        item["prompt"], completion_ids, args.max_prompt_tokens
                    )
                manager.model.train(was_training)
                log_ratio = reference - log_probs
                per_token = per_token + args.beta * (
                    log_ratio.exp() - 1 - log_ratio
                )
            action_loss = per_token.mean()
            (action_loss / max(available, 1)).backward()
            loss_sum += float(action_loss.detach().item())
            trained_actions += 1

    if manager.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(manager.device))
    global_signal = int(sum_cpu_scalars([float(local_signal)], num_processes)[0])
    if global_signal == 0:
        return {
            "did_step": False, "loss": float("nan"), "grad_norm": 0.0,
            "trained_actions": 0, "reward_std": reward_std,
        }
    average_gradients(manager.model, num_processes)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters(manager.model), args.grad_clip_norm
    )
    optimizer.step()
    scheduler.step()
    return {
        "did_step": True,
        "loss": loss_sum / trained_actions if trained_actions else 0.0,
        "grad_norm": float(grad_norm.detach().item()),
        "trained_actions": trained_actions,
        "reward_std": reward_std,
    }


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def cpu_barrier(num_processes: int) -> None:
    sum_cpu_scalars([0.0], num_processes)


def save_checkpoint(
    directory: Path,
    manager: TextModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    trainer_state: dict[str, Any],
) -> None:
    if accelerator.is_main_process:
        directory.mkdir(parents=True, exist_ok=True)
        manager.model.save_pretrained(directory)
        manager.tokenizer.save_pretrained(directory)
        torch.save({
            **trainer_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, directory / "trainer_state.pt")
    cpu_barrier(accelerator.num_processes)
    torch.save(
        capture_rng_state(), directory / f"rng_rank_{accelerator.process_index}.pt"
    )
    cpu_barrier(accelerator.num_processes)


def load_resume_state(
    directory: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
) -> dict[str, Any]:
    state_path = directory / "trainer_state.pt"
    rng_path = directory / f"rng_rank_{accelerator.process_index}.pt"
    if not state_path.is_file() or not rng_path.is_file():
        raise ValueError(
            f"{directory} is not a complete checkpoint; trainer_state.pt and "
            f"rng_rank_{accelerator.process_index}.pt are required"
        )
    state = torch.load(state_path, map_location=accelerator.device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return state


def validate_epoch(
    manager: TextModel,
    answer: TextModel,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    started = perf_counter()

    def report(message: str) -> None:
        print(
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"manager-validation rank=0 epoch={epoch} "
            f"total_elapsed_s={perf_counter()-started:.1f} {message}",
            flush=True,
        )

    report("event=start")
    states = read_json_or_jsonl(args.validation_states)
    timelines = read_json_or_jsonl(args.validation_timelines)
    unique_states = len({
        (row["dialogue_id"], int(row["target_turn"])) for row in states
    })
    if len(states) != 81 or unique_states != 66:
        raise ValueError(
            f"Validation contract failed: expected 81 QA / 66 states, got "
            f"{len(states)} / {unique_states}"
        )
    if (
        len(timelines) != 1
        or len(timelines[0].get("turns", [])) != 369
        or len(timelines[0].get("questions", [])) != 81
    ):
        raise ValueError("Validation timeline contract failed: expected 1 dialogue, 369 turns, 81 QA")

    def manager_generate(prompt: str) -> str:
        return manager.generate(
            prompt, args.max_new_tokens, 0.0, args.max_prompt_tokens
        )[0]

    def answer_generate(
        question: str, bank: list[dict[str, Any]], participants: list[str]
    ) -> str:
        return answer_from_bank(answer, question, bank, participants, args)

    report(f"stage=single_step event=start qa={len(states)} unique_states={unique_states}")
    single = evaluate_single_step(
        states, manager_generate, answer_generate, args.manager_top_k,
        progress=report,
    )
    report(
        f"stage=single_step event=done em={single['summary']['em']:.4f} "
        f"f1={single['summary']['f1']:.4f}"
    )
    report(
        f"stage=roll_forward event=start dialogues={len(timelines)} "
        f"turns={sum(len(row['turns']) for row in timelines)} "
        f"qa={sum(len(row['questions']) for row in timelines)}"
    )
    roll_forward = evaluate_roll_forward(
        timelines, manager_generate, answer_generate, args.manager_top_k,
        progress=report,
    )
    result = {"epoch": epoch, "single_step": single, "roll_forward": roll_forward}
    output = Path(args.output_dir) / "evaluation" / f"epoch_{epoch}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report(f"event=done output={output}")
    print(
        f"Manager validation epoch {epoch}: "
        f"single EM={single['summary']['em']:.4f} F1={single['summary']['f1']:.4f}; "
        f"roll-forward EM={roll_forward['summary']['em']:.4f} "
        f"F1={roll_forward['summary']['f1']:.4f} "
        f"invalid_turns={roll_forward['summary']['invalid_turns']}",
        flush=True,
    )
    return result


def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    args.device = str(accelerator.device)
    set_seed(args.seed)
    initialize_collectives(args.device, accelerator.num_processes, enable_cpu_sync=True)

    resume_dir = Path(args.resume_from) if args.resume_from else None
    manager = TextModel(
        args.manager_model, args.device, trainable=True,
        adapter_path=str(resume_dir) if resume_dir else None,
    )
    answer = TextModel(
        args.answer_model, args.device, trainable=False,
        adapter_path=args.answer_adapter,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(manager.model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / max(args.warmup_steps, 1), 1.0),
    )

    resume_epoch = 1
    next_group_index = 0
    global_step = 0
    best_f1 = -1.0
    best_em = -1.0
    best_epoch = 0
    baseline_f1 = -1.0
    baseline_em = -1.0
    resume_epoch_totals: dict[str, float] | None = None
    resume_epoch_optimizer_steps = 0
    if resume_dir:
        state = load_resume_state(resume_dir, optimizer, scheduler, accelerator)
        resume_epoch = int(state["resume_epoch"])
        next_group_index = int(state["next_group_index"])
        global_step = int(state["global_step"])
        best_f1 = float(state.get("best_f1", -1.0))
        best_em = float(state.get("best_em", -1.0))
        best_epoch = int(state.get("best_epoch", 0))
        baseline_f1 = float(state["baseline_f1"])
        baseline_em = float(state["baseline_em"])
        resume_epoch_totals = state.get("epoch_totals")
        resume_epoch_optimizer_steps = int(state.get("epoch_optimizer_steps", 0))

    states = read_training_states(args.data_path)
    progress_log = getattr(args, "progress_log", None) or str(Path(args.output_dir) / "training_progress.jsonl")
    append_progress(progress_log, "start", component="manager", epochs=args.epochs,
                    states=len(states), world_size=accelerator.num_processes,
                    data_path=args.data_path)
    unique_targets = len({
        (row["dialogue_id"], int(row["target_turn"])) for row in states
    })
    if len(states) != 150 or unique_targets != 113:
        raise ValueError(
            f"Training contract failed: expected 150 QA / 113 states, got "
            f"{len(states)} / {unique_targets}"
        )
    local_states = shard_rows(states, accelerator.process_index, accelerator.num_processes)
    scheduled = len(local_states) * accelerator.num_processes
    if accelerator.is_main_process:
        print(
            "Manager single-step config: "
            f"world_size={accelerator.num_processes}, QA={len(states)}, "
            f"unique_states={unique_targets}, steps_per_rank={len(local_states)}, "
            f"scheduled_groups={scheduled}, padding={scheduled-len(states)}, "
            f"G={args.num_generations}, reward=continuous_token_f1, "
            f"manager_temperature={args.temperature}, answer_temperature=0.0, "
            f"lr={args.learning_rate:.2e}, warmup_steps={args.warmup_steps}, "
            f"weight_decay={args.weight_decay}, beta={args.beta}, "
            f"clip={args.clip_epsilon}, dropout=0.0, "
            f"response_tokens={args.max_new_tokens}, prompt_tokens={args.max_prompt_tokens}",
            flush=True,
        )

    if not resume_dir:
        if accelerator.is_main_process:
            baseline = validate_epoch(manager, answer, args, 0)
            baseline_metric = baseline["roll_forward"]["summary"]
            baseline_f1 = float(baseline_metric["f1"])
            baseline_em = float(baseline_metric["em"])
            print(
                f"Manager epoch-0 baseline: roll-forward EM={baseline_em:.4f} "
                f"F1={baseline_f1:.4f}", flush=True,
            )
        cpu_barrier(accelerator.num_processes)

    for epoch in range(resume_epoch, args.epochs + 1):
        start_index = next_group_index if epoch == resume_epoch else 0
        progress = tqdm(
            range(start_index, len(local_states)),
            desc=f"manager-single-step epoch={epoch}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )
        totals = {
            "groups": 0.0, "signal_groups": 0.0, "step_votes": 0.0,
            "reward_sum": 0.0, "reward_count": 0.0, "reward_nonzero": 0.0,
            "em_sum": 0.0, "valid": 0.0, "invalid": 0.0, "actions": 0.0,
            "loss_sum": 0.0, "loss_count": 0.0, "grad_sum": 0.0,
            "grad_max": 0.0,
        }
        if epoch == resume_epoch and start_index and resume_epoch_totals:
            totals.update({key: float(value) for key, value in resume_epoch_totals.items()})
        epoch_optimizer_steps = (
            resume_epoch_optimizer_steps if epoch == resume_epoch and start_index else 0
        )
        for group_index in progress:
            state = local_states[group_index]
            rollouts = rollout_group(manager, answer, state, args)
            update = update_policy(
                manager, optimizer, scheduler, rollouts, args, accelerator.num_processes
            )
            if update["did_step"]:
                global_step += 1
                epoch_optimizer_steps += 1
            local = [
                1.0,
                float(update["reward_std"] > args.variance_epsilon),
                float(update["did_step"]),
                sum(item["reward"] for item in rollouts),
                float(len(rollouts)),
                sum(item["reward"] > 0 for item in rollouts),
                sum(item["em"] for item in rollouts),
                sum(item["valid"] for item in rollouts),
                sum(not item["valid"] for item in rollouts),
                sum(item["actions"] for item in rollouts),
                0.0 if math.isnan(update["loss"]) else update["loss"],
                0.0 if math.isnan(update["loss"]) else 1.0,
                update["grad_norm"],
            ]
            values = sum_cpu_scalars(local, accelerator.num_processes)
            keys = [
                "groups", "signal_groups", "step_votes", "reward_sum", "reward_count",
                "reward_nonzero", "em_sum", "valid", "invalid", "actions",
                "loss_sum", "loss_count", "grad_sum",
            ]
            if accelerator.is_main_process:
                for key, value in zip(keys, values):
                    totals[key] += value
                totals["grad_max"] = max(
                    totals["grad_max"], values[-1] / accelerator.num_processes
                )
            global_reward = values[3] / max(values[4], 1.0)
            if accelerator.is_main_process:
                append_progress(progress_log, "group_done", component="manager", epoch=epoch,
                                group=group_index + 1, total_groups=len(local_states),
                                global_step=global_step, reward=global_reward,
                                reward_std=update["reward_std"], did_step=update["did_step"],
                                loss=None if math.isnan(update["loss"]) else update["loss"])
            progress.set_postfix(
                reward=f"{global_reward:.4f}",
                reward_std=f"{update['reward_std']:.4f}",
                signal=f"{int(values[1])}/{accelerator.num_processes}",
                valid=f"{int(values[7])}/{int(values[4])}",
                loss="nan" if math.isnan(update["loss"]) else f"{update['loss']:.4f}",
                grad=f"{values[-1] / accelerator.num_processes:.3e}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if args.save_every_groups and (group_index + 1) % args.save_every_groups == 0:
                checkpoint = (
                    Path(args.output_dir) / "checkpoints" /
                    f"epoch_{epoch}_group_{group_index + 1}"
                )
                save_checkpoint(
                    checkpoint, manager, optimizer, scheduler, accelerator,
                    {
                        "completed_epoch": epoch - 1,
                        "resume_epoch": epoch,
                        "next_group_index": group_index + 1,
                        "global_step": global_step,
                        "best_f1": best_f1,
                        "best_em": best_em,
                        "best_epoch": best_epoch,
                        "baseline_f1": baseline_f1,
                        "baseline_em": baseline_em,
                        "epoch_totals": totals,
                        "epoch_optimizer_steps": epoch_optimizer_steps,
                    },
                )

        if accelerator.is_main_process:
            append_progress(progress_log, "epoch_done", component="manager", epoch=epoch,
                            optimizer_steps=epoch_optimizer_steps,
                            reward=totals["reward_sum"] / max(totals["reward_count"], 1),
                            signal_groups=int(totals["signal_groups"]), groups=int(totals["groups"]))
            print(
                f"Manager epoch {epoch} summary: "
                f"optimizer_steps={epoch_optimizer_steps}, global_step={global_step}, "
                f"loss={totals['loss_sum']/max(totals['loss_count'],1):.6f}, "
                f"reward={totals['reward_sum']/max(totals['reward_count'],1):.6f}, "
                f"strict_em={totals['em_sum']/max(totals['reward_count'],1):.6f}, "
                f"reward_signal_groups={int(totals['signal_groups'])}/{int(totals['groups'])}, "
                f"valid_generations={int(totals['valid'])}/{int(totals['reward_count'])}, "
                f"invalid_generations={int(totals['invalid'])}, actions={int(totals['actions'])}, "
                f"grad_norm_mean={totals['grad_sum']/max(totals['groups'],1):.6e}, "
                f"grad_norm_max={totals['grad_max']:.6e}, "
                f"lr={optimizer.param_groups[0]['lr']:.6e}",
                flush=True,
            )

        validation = None
        stop_training = False
        if accelerator.is_main_process:
            validation = validate_epoch(manager, answer, args, epoch)
            metric = validation["roll_forward"]["summary"]
            signal_ratio = totals["signal_groups"] / max(totals["groups"], 1.0)
            epoch_gate_passed = not (
                epoch == 1 and (
                    signal_ratio < args.epoch1_min_signal_ratio
                    or metric["f1"] < baseline_f1
                )
            )
            stop_training = not epoch_gate_passed
            is_best = epoch_gate_passed and (
                metric["f1"] > best_f1
                or (metric["f1"] == best_f1 and metric["em"] > best_em)
            )
            if is_best:
                best_f1, best_em, best_epoch = metric["f1"], metric["em"], epoch
            if stop_training:
                print(
                    "EPOCH-1 GATE FAILED: "
                    f"signal_ratio={signal_ratio:.4f} "
                    f"(required {args.epoch1_min_signal_ratio:.4f}), "
                    f"roll_forward_f1={metric['f1']:.4f} "
                    f"(baseline {baseline_f1:.4f}); stopping training.",
                    flush=True,
                )
        else:
            is_best = False
        cpu_barrier(accelerator.num_processes)

        epoch_dir = Path(args.output_dir) / f"epoch_{epoch}"
        save_checkpoint(
            epoch_dir, manager, optimizer, scheduler, accelerator,
            {
                "completed_epoch": epoch,
                "resume_epoch": epoch + 1,
                "next_group_index": 0,
                "global_step": global_step,
                "best_f1": best_f1,
                "best_em": best_em,
                "best_epoch": best_epoch,
                "baseline_f1": baseline_f1,
                "baseline_em": baseline_em,
                "validation_summary": validation and {
                    "single_step": validation["single_step"]["summary"],
                    "roll_forward": validation["roll_forward"]["summary"],
                },
            },
        )
        if accelerator.is_main_process and is_best:
            (Path(args.output_dir) / "best_checkpoint.json").write_text(
                json.dumps({
                    "epoch": epoch,
                    "path": str(epoch_dir),
                    "roll_forward_f1": best_f1,
                    "roll_forward_em": best_em,
                }, indent=2),
                encoding="utf-8",
            )
        if accelerator.is_main_process and stop_training:
            (Path(args.output_dir) / "training_gate_failed.json").write_text(
                json.dumps({
                    "epoch": epoch,
                    "reason": "epoch_1_acceptance_gate",
                    "signal_ratio": totals["signal_groups"] / max(totals["groups"], 1.0),
                    "required_signal_ratio": args.epoch1_min_signal_ratio,
                    "roll_forward_f1": validation["roll_forward"]["summary"]["f1"],
                    "baseline_f1": baseline_f1,
                }, indent=2),
                encoding="utf-8",
            )
        stop_votes = sum_cpu_scalars(
            [float(accelerator.is_main_process and stop_training)],
            accelerator.num_processes,
        )[0]
        if stop_votes:
            break
        next_group_index = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", default="data/stage0/manager_states_train.jsonl",
        help="Stage-0 single-step training states",
    )
    parser.add_argument(
        "--validation-states",
        default="data/stage0/manager_states_validation.jsonl",
    )
    parser.add_argument(
        "--validation-timelines",
        default="data/stage0/dialogue_timelines_validation.json",
    )
    parser.add_argument(
        "--precheck-path", default="data/stage0/precheck_train_f1.json",
        help="Passed Stage-0.5 counterfactual report",
    )
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--answer-adapter")
    parser.add_argument("--output-dir", default="output/memory_r1_manager")
    parser.add_argument("--resume-from", help="Complete checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--variance-epsilon", type=float, default=1e-12)
    parser.add_argument("--epoch1-min-signal-ratio", type=float, default=0.15)
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--answer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--save-every-groups", type=int, default=0,
        help="Write a resumable mid-epoch checkpoint every N local groups; 0 disables",
    )
    parser.add_argument("--progress-log", help="Append-only JSONL progress log; defaults to output-dir/training_progress.jsonl")
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 5:
        parser.error("--epochs must be between 1 and 5")
    if args.num_generations < 2:
        parser.error("--num-generations must be at least 2")
    for path in (
        args.data_path, args.validation_states, args.validation_timelines,
        args.precheck_path,
    ):
        if not Path(path).is_file():
            parser.error(f"required data file does not exist: {path}")
    precheck = json.loads(Path(args.precheck_path).read_text(encoding="utf-8"))["summary"]
    if precheck.get("training_reward_rule") != "continuous_token_f1_unique_overlap":
        parser.error("precheck uses a stale reward rule; rescore it before training")
    if not precheck.get("gate_passed"):
        parser.error("Stage-0.5 counterfactual gate failed; refusing to train")
    try:
        train(args)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
