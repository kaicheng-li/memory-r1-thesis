"""Per-question trajectory credit for the controlled B/C comparison.

Rewards are [question][trajectory][action]. Never average question rewards
before group normalization. Invalid actions get return zero AFTER computing
return-to-go: earlier valid actions still receive the trajectory's outcome.
"""
from __future__ import annotations

import torch
from memfactory.oracle_reward import estimate_advantages


def question_credit(trajectories, question_rewards, gamma=1.0, epsilon=1e-12, device="cpu"):
    contributions = {}
    groups = []
    question_count = len(question_rewards)
    for q_index, rewards in enumerate(question_rewards):
        if len(rewards) != len(trajectories):
            raise ValueError("Each question must score every trajectory")
        returns = []
        for trajectory, row in zip(trajectories, rewards):
            actions = trajectory["actions"]
            if len(row) != len(actions):
                raise ValueError("Reward/action alignment mismatch")
            values, running = [0.0] * len(row), 0.0
            for j in reversed(range(len(row))):
                running = float(row[j]) + gamma * running
                values[j] = running if actions[j].get("valid", True) else 0.0
            returns.append(values)
        positions = sorted({a["turn_position"] for tr in trajectories for a in tr["actions"]})
        eligible = []
        for position in positions:
            active = [(i, j) for i, tr in enumerate(trajectories)
                      for j, a in enumerate(tr["actions"]) if a["turn_position"] == position]
            values = torch.tensor([returns[i][j] for i, j in active], dtype=torch.float32, device=device)
            advantages, skip = estimate_advantages(values, "grpo", epsilon=epsilon)
            groups.append({"question_index": q_index, "turn_position": position,
                           "returns": values.tolist(), "advantages": advantages.tolist(), "skipped": skip})
            if not skip:
                for (i, j), advantage in zip(active, advantages):
                    if trajectories[i]["actions"][j]["completion_ids"].numel():
                        eligible.append(((i, j), advantage))
        # A zero-signal question contributes zero, rather than increasing the
        # weight of another question. Each question averages its active actions.
        for key, advantage in eligible:
            contributions.setdefault(key, []).append(
                (q_index, advantage, 1.0 / (question_count * len(eligible))))
    return contributions, groups


def clipped_surrogate(log_probs, old_log_probs, advantage, clip_epsilon):
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    return -torch.minimum(ratio * advantage, clipped * advantage).mean()
