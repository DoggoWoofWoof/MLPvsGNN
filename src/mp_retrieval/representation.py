"""Mechanistic diagnostics for message-passing representations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _sample_rows(x: torch.Tensor, limit: int, seed: int) -> torch.Tensor:
    x = x.detach().cpu().float()
    if x.shape[0] <= limit:
        return x
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return x[torch.randperm(x.shape[0], generator=generator)[:limit]]


def effective_rank(x: torch.Tensor, *, max_rows: int = 10_000, seed: int = 0) -> float:
    """Entropy-based effective rank of centered representations."""

    sample = _sample_rows(x, max_rows, seed)
    centered = sample - sample.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if float(total) == 0.0:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def stable_rank(x: torch.Tensor, *, max_rows: int = 10_000, seed: int = 0) -> float:
    sample = _sample_rows(x, max_rows, seed)
    centered = sample - sample.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    if singular_values.numel() == 0 or float(singular_values.max()) == 0.0:
        return 0.0
    return float(singular_values.square().sum() / singular_values.max().square())


def cosine_concentration(
    x: torch.Tensor,
    *,
    num_pairs: int = 100_000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean and dispersion of randomly sampled off-diagonal cosine values."""

    features = F.normalize(x.detach().cpu().float(), dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left = torch.randint(features.shape[0], (num_pairs,), generator=generator)
    right = torch.randint(features.shape[0], (num_pairs,), generator=generator)
    valid = left != right
    values = (features[left[valid]] * features[right[valid]]).sum(dim=-1)
    return {
        "cosine_mean": float(values.mean()) if values.numel() else 1.0,
        "cosine_std": float(values.std(unbiased=False)) if values.numel() else 0.0,
        "cosine_p95": float(torch.quantile(values, 0.95)) if values.numel() else 1.0,
    }


def normalized_dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Mean squared edge difference divided by mean representation energy."""

    features = x.detach().cpu().float()
    src, dst = edge_index.cpu()
    numerator = (features[src] - features[dst]).square().sum(dim=-1).mean()
    denominator = features.square().sum(dim=-1).mean().clamp_min(1e-12)
    return float(numerator / denominator)


def hub_amplification(x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, float]:
    """Correlation/slope between degree and representation norm."""

    num_nodes = x.shape[0]
    degree = torch.bincount(edge_index.cpu().flatten(), minlength=num_nodes).float()
    norms = x.detach().cpu().float().norm(dim=-1)
    degree_centered = degree - degree.mean()
    norm_centered = norms - norms.mean()
    covariance = (degree_centered * norm_centered).mean()
    correlation = covariance / (
        degree_centered.std(unbiased=False) * norm_centered.std(unbiased=False)
    ).clamp_min(1e-12)
    slope = covariance / degree_centered.square().mean().clamp_min(1e-12)
    return {"degree_norm_correlation": float(correlation), "degree_norm_slope": float(slope)}


def representation_diagnostics(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    seed: int = 0,
) -> dict[str, float]:
    result = {
        "effective_rank": effective_rank(x, seed=seed),
        "effective_rank_fraction": effective_rank(x, seed=seed) / max(min(x.shape), 1),
        "stable_rank": stable_rank(x, seed=seed),
        "dirichlet_energy": normalized_dirichlet_energy(x, edge_index),
        "mean_norm": float(x.detach().float().norm(dim=-1).mean()),
        "finite_fraction": float(torch.isfinite(x).float().mean()),
    }
    result.update(cosine_concentration(x, seed=seed))
    result.update(hub_amplification(x, edge_index))
    return result


def gradient_health(model: torch.nn.Module) -> dict[str, dict[str, float | bool]]:
    """Per-parameter gradient norms; used to fail runs with dead GNN layers."""

    report: dict[str, dict[str, float | bool]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        report[name] = {
            "present": grad is not None,
            "finite": bool(grad is not None and torch.isfinite(grad).all()),
            "norm": 0.0 if grad is None else float(grad.detach().norm()),
        }
    return report


def assert_message_passing_gradients(model: torch.nn.Module) -> None:
    """Raise if any trainable convolution parameter has no usable gradient."""

    bad = []
    for name, values in gradient_health(model).items():
        if "conv" in name.lower() and (
            not values["present"] or not values["finite"] or math.isclose(float(values["norm"]), 0.0)
        ):
            bad.append(name)
    if bad:
        raise RuntimeError(f"Dead or invalid message-passing gradients: {bad}")

