"""Single-step advantages; reference scores are never used to filter QA."""
import math
import torch


def advantages(rewards, mode="grpo", reference=0.0, scale=1.0, epsilon=1e-12):
    if rewards.ndim != 1 or rewards.numel() < 2 or not torch.isfinite(rewards).all():
        raise ValueError("Expected at least two finite scalar candidate rewards")
    if not all(math.isfinite(v) for v in (reference, scale, epsilon)) or scale <= 0 or epsilon < 0:
        raise ValueError("Invalid reference/scale/epsilon")
    if not 0 <= reference <= 1 or torch.any((rewards < 0) | (rewards > 1)):
        raise ValueError("F1 and reference must lie in [0,1]")
    std = float(rewards.std(unbiased=False).item())
    if mode == "grpo":
        signal = std > epsilon
        result = (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6) if signal else torch.zeros_like(rewards)
    elif mode in ("fixed-zero", "raw-reference"):
        residual = rewards - (reference if mode == "raw-reference" else 0.0)
        signal = float(residual.abs().max()) > epsilon
        result = residual / scale
    else:
        raise ValueError(f"Unknown advantage mode: {mode}")
    return result.detach(), signal, std


def quadrant(reference_em, candidate_em):
    return f"reference_{'correct' if reference_em else 'wrong'}__memory_{'correct' if candidate_em else 'wrong'}"
