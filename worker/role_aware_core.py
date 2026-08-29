"""Pure, testable primitives for the HALSP role-aware Focus screen.

This module contains no job dispatch, network access, dataset access, or shell
execution.  The training adapter supplies signed gate derivatives gathered by
the reviewed dense scoring probe; these functions turn them into deterministic
scores, conflict diagnostics, and bounded modulation coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


EPS = 1e-12


@dataclass(frozen=True)
class RoleScores:
    joint_taylor: torch.Tensor
    structural_taylor: torch.Tensor
    role_aware_taylor: torch.Tensor
    conflict: float


def _vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def calculate_role_scores(
    entry: torch.Tensor,
    inner: torch.Tensor,
    exit: torch.Tensor,
    depthwise: torch.Tensor,
) -> RoleScores:
    """Return the three preregistered scores from signed mean-loss derivatives.

    ``inner`` and ``depthwise`` must already be summed *signed* over blocks and
    probe batches.  Absolute-per-block values are diagnostics only and must not
    be passed here.
    """

    values = {
        "entry": _vector("entry", entry),
        "inner": _vector("inner", inner),
        "exit": _vector("exit", exit),
        "depthwise": _vector("depthwise", depthwise),
    }
    shapes = {tuple(value.shape) for value in values.values()}
    if len(shapes) != 1:
        raise ValueError("all role derivative vectors must have the same shape")

    tied_sum = values["entry"] + values["inner"] + values["exit"]
    joint = (tied_sum + values["depthwise"]).abs()
    structural = tied_sum.abs() + values["depthwise"].abs()
    role_aware = (
        values["entry"].abs()
        + values["inner"].abs()
        + values["exit"].abs()
        + values["depthwise"].abs()
    )
    denominator = (
        values["entry"].abs()
        + values["inner"].abs()
        + values["exit"].abs()
    ).sum()
    if float(denominator) <= EPS:
        raise ValueError("entry/inner/exit derivatives have no measurable signal")
    conflict = 1.0 - float(tied_sum.abs().sum() / (denominator + EPS))
    conflict = min(1.0, max(0.0, conflict))
    return RoleScores(joint, structural, role_aware, conflict)


def exact_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Select exactly K using score-descending, channel-index-ascending ties."""

    scores = _vector("scores", scores)
    if not 1 <= k <= scores.numel():
        raise ValueError("k must be between one and the channel count")
    # Stable descending sort preserves the original ascending index order for
    # equal scores.  CPU and CUDA both support stable argsort in supported torch.
    return torch.argsort(scores, descending=True, stable=True)[:k]


def selected_swap_fraction(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    """q = 1 - |A intersection B| / K for two exact-K selected sets."""

    left = torch.as_tensor(left, dtype=torch.long).flatten()
    right = torch.as_tensor(right, dtype=torch.long).flatten()
    if left.numel() != k or right.numel() != k:
        raise ValueError("both selections must contain exactly k indices")
    if torch.unique(left).numel() != k or torch.unique(right).numel() != k:
        raise ValueError("selected indices must be unique")
    overlap = torch.isin(left, right).sum().item()
    return 1.0 - float(overlap) / float(k)


def role_alphas(two_dof: torch.Tensor) -> torch.Tensor:
    """Map two unconstrained values/channel to three positive mean-one alphas.

    The equilateral basis has column sums zero. Smooth radial normalization
    bounds the contrast while retaining a non-zero Jacobian at the origin, so
    every coefficient remains in [0.75, 1.25] and u=0 is exactly identity.
    Output shape is ``[..., 3]`` in entry/inner/exit order.
    """

    if not isinstance(two_dof, torch.Tensor) or two_dof.shape[-1] != 2:
        raise ValueError("two_dof must have final dimension two")
    root3_over_2 = two_dof.new_tensor(3.0).sqrt() / 2.0
    basis = two_dof.new_tensor(
        [[1.0, 0.0], [-0.5, float(root3_over_2)], [-0.5, -float(root3_over_2)]]
    )
    raw = two_dof @ basis.T
    bounded = raw / torch.sqrt(1.0 + raw.square().sum(dim=-1, keepdim=True))
    return 1.0 + 0.25 * bounded


def full_support_momentum_update(
    previous: torch.Tensor | None,
    probe_gradient: torch.Tensor,
    beta: float = 0.9,
) -> torch.Tensor:
    """Checkpointable probe-only EMA; never aliases optimizer momentum."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if not torch.isfinite(probe_gradient).all():
        raise ValueError("probe gradient contains NaN or Inf")
    if previous is None:
        previous = torch.zeros_like(probe_gradient)
    if previous.shape != probe_gradient.shape:
        raise ValueError("probe momentum and gradient shapes differ")
    return previous.mul(beta).add(probe_gradient)


def current_form_score(weight: torch.Tensor, probe_momentum: torch.Tensor) -> torch.Tensor:
    """Historical |W*m| row form supplied with fair full-support information."""

    if weight.shape != probe_momentum.shape or weight.ndim < 2:
        raise ValueError("weight and probe_momentum must share a row-major shape")
    return (weight * probe_momentum).abs().flatten(1).mean(1)


def material_checkpoint(stage_metrics: list[Mapping[str, float]]) -> bool:
    """Frozen gate across the two sparse HALSP stages (layer3/layer4).

    The pinned architecture deliberately keeps layer1/layer2 dense.  Their role
    metrics may be logged, but they cannot enter a Focus-selection gate.
    """

    if len(stage_metrics) != 2:
        raise ValueError("exactly the two sparse-stage metrics are required")
    conflicts = torch.tensor([float(row["conflict"]) for row in stage_metrics])
    swaps = torch.tensor([float(row["swap_fraction"]) for row in stage_metrics])
    median_conflict = float(conflicts.mean())
    median_swap = float(swaps.mean())
    return bool(
        median_conflict >= 0.25
        and median_swap >= 0.111
        and int((conflicts >= 0.25).sum()) >= 1
    )


def open_modulation_gate(seed_checkpoint_flags: Mapping[int, list[bool]]) -> bool:
    """Both seeds: >=3/5 material checkpoints and final checkpoint material."""

    if set(seed_checkpoint_flags) != {0, 1}:
        raise ValueError("the screening gate requires exactly seeds 0 and 1")
    for flags in seed_checkpoint_flags.values():
        if len(flags) != 5 or sum(bool(flag) for flag in flags) < 3 or not flags[-1]:
            return False
    return True
