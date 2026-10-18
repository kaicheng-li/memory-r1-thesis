"""Controlled sequential Manager training for A/B/C experiments.

B/C replay cached facts from an empty bank over each full causal prefix.
They share per-question GRPO and invalid-action return masking; only evidence
weight differs. The frozen base reader scores terminal QA. Answer training
later uses full-dialogue replay, matching baseline-lora's downstream protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_retriever import build_answer_input_from_memories
from memfactory.modules.memory_updater import build_manager_input
from memfactory.lora import load_lora_model, trainable_parameters
from memfactory.chat import DEFAULT_MAX_PROMPT_TOKENS, tokenize_chat_prompt
from memfactory.locomo_split import SPLIT_DIALOGUE_IDS
from memfactory.run_metadata import write_run_metadata
from memfactory.sequential_validation import load_validation_timelines, evaluate_roll_forward
from evaluation.metrics import normalize_text, exact_match as exact_match_reward, f1_score as answer_reward
from memfactory.oracle_reward import (safe_answer, extract_answer, gold_text, evaluate_controls, ContextBudgetError,
                                      estimate_advantages, append_jsonl)
from memfactory.memory_runtime import apply_decisions, parse_manager_output, retrieve, retrieve_per_speaker
from memfactory.progress import append_progress


def read_rows(path: str) -> list[dict[str, Any]]:
    """Read a JSON or JSONL dataset into a list of rows."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


# ---------------------------------------------------------------------------
# Algorithm 1 tuple loading
# ---------------------------------------------------------------------------

def normalize_tuples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Algorithm 1 rows -> (dialogue_turns, temporal_memory_bank, current_turn, linked_questions)."""
    normalized = []
    for row in rows:
        if row.get("initial_bank") or row.get("initial_memory_bank") or row.get("temporal_memory_bank"):
            raise ValueError("Formal fullprefix input starts empty; nonempty initial banks require a separately specified experiment")
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        normalized.append({
            "dialogue_id": str(row["dialogue_id"]),
            "split": row.get("split", metadata.get("split")),
            "turn_index": row["turn_index"],
            "participants": row["participants"],
            "temporal_memory_bank": row["temporal_memory_bank"],
            "dialogue_turns": row["dialogue_turns"],
            "current_turn": row["current_turn"],
            "linked_questions": row["linked_questions"],
        })
    return normalized


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def evidence_ids(question: dict[str, Any]) -> set[str]:
    """Return source turn ids supplied by LoCoMo QA annotations."""
    return set(question["evidence"])


def memory_source_ids(memory: list[dict[str, Any]]) -> set[str]:
    """Collect provenance ids without treating updated text as new evidence."""
    return {source for entry in memory for source in entry.get("source_turn_ids", [])}


def potential_shaping_rewards(
    potentials: list[float], gamma: float, terminal_potential: str = "observed",
    initial_potential: float = 0.0,
) -> list[float]:
    """Potential differences; observed terminal coverage is an auxiliary goal.

    With a zero terminal potential the discounted sum is -initial_potential.
    That boundary condition alone does not guarantee better GRPO credit.
    """
    if terminal_potential not in {"observed", "zero"}:
        raise ValueError("terminal_potential must be observed or zero")
    values = list(potentials)
    if values and terminal_potential == "zero":
        values[-1] = 0.0
    previous = initial_potential
    rewards = []
    for current in values:
        rewards.append(gamma * current - previous)
        previous = current
    return rewards


def coverage_rewards(potentials, gamma=1.0, mode="dense", terminal_potential="observed"):
    if mode == "dense":
        return potential_shaping_rewards(potentials, gamma, terminal_potential)
    if mode != "terminal" or gamma != 1.0 or terminal_potential != "observed":
        raise ValueError("Terminal-only coverage requires gamma=1 and observed boundary")
    return [0.0] * (len(potentials) - 1) + [potentials[-1]] if potentials else []


def evidence_potential(
    question: dict[str, Any],
    memory: list[dict[str, Any]],
    answer_top_k: int,
    participants: list[str],
    available_evidence: set[str],
) -> float:
    """Measure how much annotated evidence is stored and retrievable.

    This is training-only privileged information. The Answer Agent still sees
    only the retrieved memory, never the original dialogue or evidence labels.
    """
    # Turn-origin coverage proxy from the thesis outline. No semantic support
    # claim, and no additional retrieval objective. New evidence can expand
    # the denominator and lower coverage without deletion.
    available = evidence_ids(question) & available_evidence
    if not available:
        return 0.0
    return len(available & memory_source_ids(memory)) / len(available)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TextModel:
    def __init__(self, model_path: str, device: str, trainable: bool, adapter_path: str | None = None,
                 max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = load_lora_model(model_path, self.device, adapter_path, trainable)
        self.model.train(trainable)
        self.trainable = trainable
        self.max_prompt_tokens = max_prompt_tokens

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, torch.Tensor]:
        """Sample a completion; returns (decoded text, raw sampled token ids).

        The token ids are the exact sequence the policy sampled and are kept
        so log_probability scores the actual rollout action instead of a
        re-tokenized reconstruction of the decoded text.
        """
        inputs = tokenize_chat_prompt(self.tokenizer, prompt, self.max_prompt_tokens).to(self.device)
        from memfactory.generation import generate_policy
        output = generate_policy(self.model, self.tokenizer, inputs, max_new_tokens, temperature)
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_length:]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            positions = (completion_ids == eos).nonzero(as_tuple=False)
            if len(positions):
                completion_ids = completion_ids[:int(positions[0]) + 1]
        completion_ids = completion_ids.clone()
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return text, completion_ids

    def log_probability(self, prompt: str, completion_ids: torch.Tensor) -> torch.Tensor:
        """Token-level log probabilities over the sampled completion, shape [L'].

        Scores the exact token ids returned by generate(), so importance
        ratios refer to the rollout action that was actually sampled.
        """
        if completion_ids.numel() == 0:
            return torch.zeros((0,), device=self.device)
        prompt_ids = tokenize_chat_prompt(self.tokenizer, prompt, self.max_prompt_tokens)["input_ids"][0]
        input_ids = torch.cat([prompt_ids.to(self.device), completion_ids.to(self.device)]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        was_training = self.model.training
        self.model.eval()
        try:
            start = max(prompt_ids.numel() - 1, 0)
            base = self.model.get_base_model() if hasattr(self.model, "peft_config") else self.model
            if hasattr(base, "model") and hasattr(base, "lm_head"):
                hidden = base.model(input_ids=input_ids, attention_mask=attention_mask,
                                    use_cache=False, return_dict=True).last_hidden_state
                completion_logits = base.lm_head(hidden[:, start:start + completion_ids.numel(), :])
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                completion_logits = outputs.logits[:, start:start + completion_ids.numel(), :]
            completion_labels = input_ids[:, start + 1:start + 1 + completion_ids.numel()]
            return F.log_softmax(completion_logits, dim=-1).gather(2, completion_labels.unsqueeze(-1)).squeeze(-1)
        finally:
            self.model.train(was_training)


def answer_with_model(
    answer_model: TextModel,
    question: str,
    memories: list[dict[str, Any]],
    participants: list[str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    if hasattr(answer_model, "tokenizer"):
        return safe_answer(answer_model, question, memories, participants, max_new_tokens,
                           getattr(answer_model, "context_overflow", "truncate"),
                           getattr(answer_model, "compression_tokens", 256))["prediction"]
    # Minimal model doubles for runtime contract tests.
    prompt = build_answer_input_from_memories(question, memories, participants)
    raw, _ = answer_model.generate(prompt, max_new_tokens, temperature=temperature)
    return extract_answer(raw)


# ---------------------------------------------------------------------------
# RL update
# ---------------------------------------------------------------------------

def apply_policy_update(manager, reference, optimizer, trajectories, rewards, args):
    from memfactory.sequential_credit import question_credit, clipped_surrogate
    question_rewards = getattr(args, "_question_rewards", None)
    if question_rewards is None:
        question_rewards = [rewards]  # single-question compatibility API
    contributions, groups = question_credit(
        trajectories, question_rewards, args.gamma,
        getattr(args, "variance_epsilon", 1e-12), manager.device)
    args._last_update = {"groups": groups, "eligible_actions": len(contributions),
                        "questions": len(question_rewards), "did_step": False}
    if not contributions:
        return float("nan")
    optimizer.zero_grad()
    loss_sum = 0.0
    for (i, j), question_terms in contributions.items():
        action = trajectories[i]["actions"][j]
        log_probs = manager.log_probability(action["prompt"], action["completion_ids"])
        kl = log_probs.new_zeros(())
        if getattr(args, "beta", 0.0):
            with torch.no_grad():
                if reference is not None:
                    ref = reference.log_probability(action["prompt"], action["completion_ids"])
                else:
                    # Same frozen base reference as baseline-lora, no third 7B.
                    with manager.model.disable_adapter():
                        ref = manager.log_probability(action["prompt"], action["completion_ids"])
            delta = ref - log_probs
            kl = args.beta * (delta.exp() - 1 - delta).mean()
        # Compute each question's surrogate BEFORE averaging; opposite-signed
        # advantages need not commute with PPO clipping. Same-sign terms do.
        action_loss = sum(weight * (clipped_surrogate(
            log_probs, action["old_token_log_probs"], advantage, args.clip_epsilon) + kl)
            for _, advantage, weight in question_terms)
        # Equal per-QA epoch weighting despite variable questions per tuple.
        action_loss = action_loss * getattr(args, "_tuple_question_weight", 1.0)
        action_loss.backward()
        loss_sum += float(action_loss.detach())
    torch.nn.utils.clip_grad_norm_(trainable_parameters(manager.model), getattr(args, "grad_clip_norm", 1.0))
    optimizer.step()
    if getattr(args, "_scheduler", None) is not None:
        args._scheduler.step()
    args._last_update["did_step"] = True
    return loss_sum


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def train_from_tuples(args, manager, reference, answer, optimizer, reward_fn, tuples,
                      start_index=0, on_group_end=None) -> None:
    """Per-question GRPO over shared fullprefix trajectories.

    Per tuple: M <- {}; replay every turn of the tuple's dialogue window
    using the turn's teacher-extracted facts (LLMExtract) — retrieve with the
    facts -> manager op -> apply; then answer the linked questions with the
    frozen Answer Agent over the resulting bank. One GRPO update per
    tuple over its num_generations trajectories. Evidence-aware potential
    differences are added to final QA reward. Their effect on credit depends
    on the terminal boundary and the return/advantage estimator.
    """
    if start_index == 0:
        args._epoch_stats = {"questions": 0, "signal_questions": 0}
    progress = tqdm(enumerate(tuples), total=len(tuples), desc=f"memory-manager-{args.algorithm}")
    for tuple_index, item in progress:
        if tuple_index < start_index:
            continue
        group_started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        questions = item["linked_questions"]
        log = {"epoch": getattr(args, "_epoch", 0), "tuple_index": tuple_index,
               "dialogue_id": item["dialogue_id"], "target_turn": item.get("turn_index"),
               "estimator": getattr(args, "advantage_estimator", "grpo"),
               "shaping_mode": getattr(args, "shaping_mode", "dense")}
        def finish():
            log["elapsed_seconds"] = round(time.monotonic() - group_started, 3)
            log["peak_cuda_allocated_bytes"] = (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
            args._epoch_stats["questions"] += len(questions)
            args._epoch_stats["signal_questions"] += len({
                g["question_index"] for g in log.get("update", {}).get("groups", []) if not g["skipped"]})
            append_jsonl(getattr(args, "group_log", None), log)
            append_progress(
                getattr(args, "progress_log", None),
                "group_done",
                epoch=log.get("epoch"), tuple_index=tuple_index,
                dialogue_id=item.get("dialogue_id"), target_turn=item.get("turn_index"),
                loss=log.get("loss"), skip_reason=log.get("skip_reason"),
                invalid_actions=sum(t.get("invalid_actions", 0)
                                     for t in log.get("trajectories", [])),
                eligible_questions=log.get("eligible_questions"),
                elapsed_seconds=log["elapsed_seconds"], timings=log.get("timings"),
                peak_cuda_allocated_bytes=log["peak_cuda_allocated_bytes"],
            )
            if on_group_end:
                on_group_end(tuple_index + 1)
        if not questions:
            log["skip_reason"] = "no_questions"
            finish()
            continue
        turns = item["dialogue_turns"]
        if getattr(args, "control_diagnostics", False):
            from memfactory.training_state import preserve_rng
            with preserve_rng():
                log["controls"] = evaluate_controls(answer, item, args, reward_fn)
            # Controls are diagnostic only: NEVER filter questions or alter advantages.
        log["eligible_questions"] = len(questions)
        log["question_ids"] = [str(q.get("question_id", "")) for q in questions]
        mean_questions = sum(len(row["linked_questions"]) for row in tuples) / len(tuples)
        args._tuple_question_weight = len(questions) / mean_questions
        log["tuple_question_weight"] = args._tuple_question_weight

        rollout_started = time.monotonic()
        trajectories = []
        for trajectory_index in range(args.num_generations):
            memory: list[dict[str, Any]] = copy.deepcopy(item.get("initial_memory_bank", []))
            actions = []
            memory_states = []
            invalid_actions = 0
            for local_index, turn in enumerate(turns):
                facts = [dict(fact) for fact in turn["facts"]
                         if isinstance(fact, dict) and str(fact.get("text", "")).strip()]
                if not facts:
                    continue
                query = " ".join(fact["text"] for fact in facts)
                old_memory = retrieve(query, memory, args.manager_top_k)
                prompt = build_manager_input(old_memory, facts)
                append_progress(getattr(args, "progress_log", None), "turn_start",
                                tuple_index=tuple_index, trajectory=trajectory_index, turn=local_index)
                turn_started = time.monotonic()
                completion, completion_ids = manager.generate(prompt, args.max_new_tokens, args.temperature)
                with torch.no_grad():
                    old_token_log_probs = manager.log_probability(prompt, completion_ids).detach()
                action = {
                    "prompt": prompt,
                    "completion_ids": completion_ids,
                    "old_token_log_probs": old_token_log_probs,
                    "turn_position": local_index,
                    "valid": True,
                }
                actions.append(action)
                try:
                    decisions = parse_manager_output(completion)
                    apply_decisions(
                        memory,
                        decisions,
                        item["dialogue_id"],
                        turn,
                        visible_memory_ids={entry["id"] for entry in old_memory},
                    )
                except (ValueError, KeyError, TypeError):
                    # Validation is atomic.  Keep the unchanged bank, then
                    # continue so earlier valid actions keep their signal.
                    action["valid"] = False
                    invalid_actions += 1
                memory_states.append(copy.deepcopy(memory))
                append_progress(getattr(args, "progress_log", None), "turn_done",
                                tuple_index=tuple_index, trajectory=trajectory_index, turn=local_index,
                                valid=action["valid"], memory_size=len(memory),
                                elapsed_seconds=round(time.monotonic() - turn_started, 3))
            trajectories.append(
                {
                    "memory": memory,
                    "actions": actions,
                    "memory_states": memory_states,
                    "invalid_actions": invalid_actions,
                }
            )

        rollout_seconds = time.monotonic() - rollout_started
        answer_started = time.monotonic()
        # [question][trajectory][action]: retain the question axis through
        # shaping, return computation, normalization and clipped losses.
        question_rewards = [[] for _ in questions]
        for trajectory_index, trajectory in enumerate(trajectories):
            trajectory["question_records"] = []
            for q_index, question in enumerate(questions):
                qa_started = time.monotonic()
                append_progress(getattr(args, "progress_log", None), "answer_start",
                                tuple_index=tuple_index, trajectory=trajectory_index,
                                question_index=q_index, question_id=str(question.get("question_id", "")))
                actions = trajectory["actions"]
                potentials = []
                for state, action in zip(trajectory["memory_states"], actions):
                    seen = {t["dia_id"] for t in turns[:action["turn_position"] + 1]}
                    potentials.append(evidence_potential(
                        question, state, args.answer_top_k, item["participants"], seen))
                shaping = coverage_rewards(potentials, args.gamma,
                    getattr(args, "shaping_mode", "dense"), getattr(args, "terminal_potential", "observed"))
                row_rewards = [args.evidence_weight * value for value in shaping]
                # One invalid action is exactly baseline's invalid candidate:
                # zero score, no Answer call. Longer trajectories still need
                # terminal QA for their earlier valid actions.
                score, result = 0.0, {"status": "invalid_single_step", "prediction": ""}
                if actions and any(a.get("valid", True) for a in actions):
                    retrieved = retrieve_per_speaker(question["question"], trajectory["memory"],
                                                    item["participants"], args.answer_top_k)
                    if hasattr(answer, "tokenizer"):
                        result = safe_answer(answer, question["question"], retrieved, item["participants"],
                            args.answer_max_new_tokens, getattr(args, "context_overflow", "truncate"),
                            getattr(args, "compression_tokens", 256))
                    else:
                        result = {"prediction": answer_with_model(answer, question["question"], retrieved,
                                  item["participants"], args.answer_max_new_tokens, args.answer_temperature)}
                    score = reward_fn(result["prediction"], gold_text(question["answer"]))
                if row_rewards:
                    row_rewards[-1] += score
                question_rewards[q_index].append(row_rewards)
                trajectory["question_records"].append({"question_id": str(question.get("question_id", "")),
                    "potentials": potentials, "shaping": shaping, "qa_score": score,
                    "answer": result, "rewards": row_rewards})
                append_progress(getattr(args, "progress_log", None), "answer_done",
                                tuple_index=tuple_index, trajectory=trajectory_index,
                                question_index=q_index, score=score,
                                elapsed_seconds=round(time.monotonic() - qa_started, 3))
        args._question_rewards = question_rewards
        # Compatibility/log summary only. This mean NEVER enters credit assignment.
        rewards = [[sum(question_rewards[q][i][t] for q in range(len(questions))) / len(questions)
                    for t in range(len(tr["actions"]))] for i, tr in enumerate(trajectories)]
        answer_seconds = time.monotonic() - answer_started
        update_started = time.monotonic()
        append_progress(getattr(args, "progress_log", None), "update_start", tuple_index=tuple_index)
        loss = apply_policy_update(manager, reference, optimizer, trajectories, rewards, args)
        append_progress(getattr(args, "progress_log", None), "update_done", tuple_index=tuple_index,
                        elapsed_seconds=round(time.monotonic() - update_started, 3),
                        loss=None if math.isnan(loss) else loss)
        log["timings"] = {"rollout_and_old_logprob_s": rollout_seconds, "answer_and_shaping_s": answer_seconds,
                          "update_s": time.monotonic() - update_started}
        log["trajectories"] = [{"invalid_actions": tr["invalid_actions"],
                                 "question_records": tr.get("question_records", []),
                                 "answers": tr.get("answers", []), "rewards": reward}
                                for tr, reward in zip(trajectories, rewards)]
        log["update"] = getattr(args, "_last_update", {})
        log["loss"] = None if math.isnan(loss) else loss
        log["skip_reason"] = "no_advantage_or_action_tokens" if math.isnan(loss) else None
        finish()
        if math.isnan(loss):
            continue
        flat_rewards = [value for row in rewards for value in row]
        progress.set_postfix(
            loss=loss,
            reward=float(torch.tensor(flat_rewards).mean()) if flat_rewards else 0.0,
            invalid=sum(row["invalid_actions"] for row in trajectories),
        )


def train(args: argparse.Namespace) -> None:
    if args.temperature != 1.0:
        raise ValueError("Manager training requires temperature=1 to match raw policy log probabilities.")
    if args.num_generations < 2:
        raise ValueError("Group-relative training requires at least two generations.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if args.answer_temperature != 0:
        raise ValueError("The fixed Answer and all controls must use greedy decoding")
    if args.advantage_estimator != "grpo" or args.control_gate != "off":
        raise ValueError("A/B/C protocol requires GRPO and diagnostic-only controls (gate off)")
    if args.gamma != 1.0:
        raise ValueError("A/B/C protocol fixes gamma=1")
    if args.shaping_mode == "terminal" and args.terminal_potential != "observed":
        raise ValueError("Terminal coverage ablation requires observed terminal potential")
    if args.warmup_steps < 0 or args.weight_decay < 0 or args.variance_epsilon < 0:
        raise ValueError("Optimizer/tolerance settings must be nonnegative")
    if args.checkpoint_every < 1 or args.compression_tokens < 1:
        raise ValueError("checkpoint-every and compression-tokens must be positive")
    set_seed(args.seed)
    validation = load_validation_timelines(args.validation_timelines) if args.validation_timelines else None
    rows = read_rows(args.data_path)
    tuples = normalize_tuples(rows)
    if args.limit_tuples is not None:
        if args.limit_tuples < 1:
            raise ValueError("limit-tuples must be positive")
        tuples = tuples[:args.limit_tuples]
    if not tuples:
        raise ValueError("No Manager training tuples were loaded.")
    if not getattr(args, "allow_nontrain_dialogues", False):
        permitted = set(SPLIT_DIALOGUE_IDS["train"])
        unexpected = sorted({
            str(item["dialogue_id"]) for item in tuples
            if item["dialogue_id"] not in permitted or item["split"] != "train"
        })
        if unexpected:
            raise ValueError(
                "Manager training data contains non-train LoCoMo dialogues: "
                + ", ".join(unexpected)
            )
    for item in tuples:
        for turn in item["dialogue_turns"]:
            if not isinstance(turn.get("facts"), list):
                raise SystemExit(
                    f"Turn without extracted facts ({item['dialogue_id']} "
                    f"turn {turn.get('turn_index', '?')}): data must be rebuilt "
                    "with build_manager_training_data.py (LLMExtract step)."
                )
    from memfactory.training_state import (resume_contract, read_training_state, restore_rng,
                                           save_training_checkpoint, preserve_rng)
    if args.group_log is None:
        args.group_log = str(Path(args.output_dir) / "groups.jsonl")
    if getattr(args, "progress_log", None) is None:
        args.progress_log = str(Path(args.output_dir) / "training_progress.jsonl")
    if (Path(args.output_dir) / "run_manifest.json").exists() and not args.resume:
        raise ValueError("Output directory already contains a run; use --resume or a new output directory")
    manifest = write_run_metadata(args, write=False)
    contract = resume_contract(manifest)
    state = read_training_state(args.resume, contract) if args.resume else None
    if state and state["epoch"] >= args.epochs:
        raise ValueError("Resume has already completed the requested epoch count")
    append_progress(args.progress_log, "model_load_start", component="manager",
                    progress_log=str(Path(args.progress_log).resolve()),
                    group_log=str(Path(args.group_log).resolve()))
    manager = TextModel(args.manager_model, args.device, trainable=True,
                        adapter_path=args.resume or args.manager_adapter, max_prompt_tokens=args.max_prompt_tokens)
    append_progress(args.progress_log, "model_load_done", component="manager")
    if args.manager_adapter:
        raise ValueError("Main A/B/C runs initialize from the base model; use --resume for continuation")
    if args.answer_adapter:
        raise ValueError("Manager training uses frozen base Answer, not a trained Answer adapter")
    reference = None  # disable_adapter() gives baseline's frozen reference
    append_progress(args.progress_log, "model_load_start", component="answer")
    answer = TextModel(args.answer_model, args.device, trainable=False, adapter_path=args.answer_adapter,
                       max_prompt_tokens=args.max_prompt_tokens)
    append_progress(args.progress_log, "model_load_done", component="answer")
    answer.context_overflow, answer.compression_tokens = args.context_overflow, args.compression_tokens
    optimizer = torch.optim.AdamW(trainable_parameters(manager.model), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min((step + 1) / max(args.warmup_steps, 1), 1.0))
    args._scheduler = scheduler
    write_run_metadata(args)
    run_started = time.monotonic()
    append_progress(args.progress_log, "start", total_tuples=len(tuples), epochs=args.epochs,
                    reward=args.reward, estimator=args.advantage_estimator,
                    shaping_mode=args.shaping_mode)
    reward_fn = exact_match_reward if args.reward == "em" else answer_reward
    def validate(epoch):
        started = time.monotonic()
        append_progress(args.progress_log, "validation_start", epoch=epoch, dialogues=len(validation))
        result = evaluate_roll_forward(
            validation,
            lambda prompt: manager.generate(prompt, args.max_new_tokens, 0.0)[0],
            lambda question, memories, participants: safe_answer(
                answer, question, memories, participants, args.answer_max_new_tokens,
                args.context_overflow, args.compression_tokens
            ),
            args, normalize_text, exact_match_reward, answer_reward,
            progress=lambda event, **fields: append_progress(args.progress_log, event, epoch=epoch, **fields),
        )
        (Path(args.output_dir) / f"validation_epoch_{epoch}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        append_progress(args.progress_log, "validation_done", epoch=epoch,
                        elapsed_seconds=round(time.monotonic() - started, 3), summary=result["summary"])
        return result["summary"]

    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        args._epoch_stats = state.get("epoch_stats") or {"questions": 0, "signal_questions": 0}
        restore_rng(state["rng"])
        start_epoch, next_tuple, best = state["epoch"], state["next_tuple"], state["best"]
        # Deduplicate by (epoch, tuple_index), respecting explicit resume events.
        append_jsonl(args.group_log, {"event": "resume", "epoch": start_epoch,
                                      "next_tuple": next_tuple, "checkpoint": args.resume})
    else:
        with preserve_rng():
            initial = validate(0) if validation else None
        start_epoch, next_tuple = 0, 0
        best = {"epoch": 0, "adapter": None,
                "reward_f1_unique_overlap": initial["f1"] if initial else None,
                "em": initial["em"] if initial else None,
                "initial_validation_f1": initial["f1"] if initial else None}
    def write_best():
        (Path(args.output_dir) / "best_checkpoint.json").write_text(
            json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    write_best()
    for epoch in range(start_epoch, args.epochs):
        args._epoch = epoch
        def checkpoint_group(cursor):
            if cursor % args.checkpoint_every == 0 and cursor < len(tuples):
                save_training_checkpoint(manager, optimizer,
                    Path(args.output_dir) / f"epoch_{epoch}_tuple_{cursor}", epoch, cursor, best, contract,
                    scheduler=scheduler, epoch_stats=args._epoch_stats)
        train_from_tuples(args, manager, reference, answer, optimizer, reward_fn, tuples,
                          start_index=next_tuple, on_group_end=checkpoint_group)
        next_tuple = 0
        stop_training = False
        if validation:
            with preserve_rng():
                metric = validate(epoch + 1)
            signal_ratio = args._epoch_stats["signal_questions"] / max(args._epoch_stats["questions"], 1)
            stop_training = epoch == 0 and (
                signal_ratio < args.epoch1_min_signal_ratio or metric["f1"] < best["initial_validation_f1"])
            if not stop_training and (metric["f1"] > best["reward_f1_unique_overlap"] or
                (metric["f1"] == best["reward_f1_unique_overlap"] and
                 (metric["em"] > best["em"] or best["epoch"] == 0))):
                best = dict(best, epoch=epoch + 1,
                            adapter=str(Path(args.output_dir).resolve() / f"epoch_{epoch + 1}"),
                            reward_f1_unique_overlap=metric["f1"], em=metric["em"])
            if stop_training:
                (Path(args.output_dir) / "stopped_early.json").write_text(json.dumps({
                    "epoch": epoch + 1, "reason": "epoch_1_acceptance_gate", "signal_ratio": signal_ratio,
                    "required_signal_ratio": args.epoch1_min_signal_ratio, "validation": metric}), encoding="utf-8")
        save_training_checkpoint(manager, optimizer, Path(args.output_dir) / f"epoch_{epoch + 1}",
                                 epoch + 1, 0, best, contract, scheduler=scheduler, epoch_stats=args._epoch_stats)
        write_best()
        append_progress(args.progress_log, "epoch_done", epoch=epoch + 1,
                        total_tuples=len(tuples),
                        elapsed_seconds=round(time.monotonic() - run_started, 3),
                        best_reward=best.get("reward_f1_unique_overlap"))
        if stop_training:
            break
    append_progress(args.progress_log, "done", epochs=args.epochs,
                    elapsed_seconds=round(time.monotonic() - run_started, 3),
                    best_reward=best.get("reward_f1_unique_overlap"))


def build_parser():
    parser = argparse.ArgumentParser(description="Controlled fullprefix Manager training (B/C).")
    parser.add_argument("--data-path", required=True, help="Algorithm 1 tuples (build_manager_training_data.py output)")
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--answer-adapter", help="Optional frozen Answer adapter; omit for frozen base Answer model.")
    parser.add_argument("--validation-timelines", help="Cached validation dialogue timelines for full-dialogue checkpoint selection.")
    parser.add_argument(
        "--allow-nontrain-dialogues",
        action="store_true",
        help="Explicitly allow a custom/non-LoCoMo training split; never use for a LoCoMo test claim.",
    )
    parser.add_argument("--output-dir", default="output/memory_r1_manager")
    parser.add_argument("--algorithm", choices=["grpo"], default="grpo")
    parser.add_argument("--reward", choices=["em", "f1"], default="f1", help="Reward on the frozen answer agent's output; defaults to baseline-lora F1. The paper uses EM.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--epoch1-min-signal-ratio", type=float, default=0.15)
    parser.add_argument("--limit-tuples", type=int, help="Explicit smoke run only; records subset in manifest")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k", type=int, default=30, help="Retrieved memories per speaker, matching Answer tuple construction.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--answer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument(
        "--variance-epsilon",
        type=float,
        default=1e-12,
        help="Skip a turn-position group when its return standard deviation is at or below this value.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount used for dense reward and return-to-go.",
    )
    parser.add_argument(
        "--terminal-potential", choices=["observed", "zero"], default="observed",
        help="observed adds a terminal coverage goal; zero cancels total shaping at gamma=1 from an empty bank.",
    )
    parser.add_argument(
        "--evidence-weight",
        type=float,
        default=0.5,
        help="Weight of evidence-aware per-turn potential shaping.",
    )
    parser.add_argument("--manager-adapter", help="Reserved for separate experiments; main A/B/C rejects this option")
    parser.add_argument("--advantage-estimator", choices=["grpo"], default="grpo")
    parser.add_argument("--control-gate", choices=["off"], default="off")
    parser.add_argument("--control-diagnostics", action="store_true", help="Also run controls for ungated GRPO")
    parser.add_argument("--context-overflow", choices=["truncate", "compress", "error"], default="truncate")
    parser.add_argument("--compression-tokens", type=int, default=256)
    parser.add_argument("--shaping-mode", choices=["dense", "terminal"], default="dense")
    parser.add_argument("--group-log", help="JSONL group audit log; defaults to output-dir/groups.jsonl")
    parser.add_argument("--progress-log", help="Flushed JSONL progress log; defaults to output-dir/training_progress.jsonl")
    parser.add_argument("--resume", help="Trusted local checkpoint directory including training_state.pt")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save optimizer/RNG/cursor every N groups")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
