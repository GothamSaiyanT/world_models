from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class V4LossParts:
    total: torch.Tensor
    base: torch.Tensor
    motion: torch.Tensor
    foreground: torch.Tensor
    delta: torch.Tensor
    ghost: torch.Tensor
    edge: torch.Tensor
    motion_fraction: torch.Tensor
    ghost_fraction: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _edge_map(x: torch.Tensor):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def balanced_motion_loss(
    prediction: torch.Tensor,
    real_current: torch.Tensor,
    real_next: torch.Tensor,
    *,
    motion_threshold: float = 0.015,
    foreground_threshold: float = 0.04,
    background_threshold: float = 0.03,
    ghost_margin: float = 0.02,
    motion_weight: float = 1.5,
    foreground_weight: float = 0.5,
    delta_weight: float = 0.75,
    ghost_weight: float = 2.0,
    edge_weight: float = 0.10,
    dilation: int = 3,
) -> V4LossParts:
    """
    Balanced motion-aware objective for V4.

    V3 fixed frozen rollouts but used strong motion/delta weights (4x and 3x),
    which could preserve old ball/paddle positions as ghost trails. V4 keeps
    moving pixels important without letting them dominate the objective and
    adds a targeted ghost penalty on pixels that should turn OFF between t and
    t+1 (bright now, background next).
    """
    squared = (prediction - real_next).pow(2)
    base = squared.mean()

    raw_motion = (
        (real_next - real_current).abs() > motion_threshold
    ).to(prediction.dtype)
    if dilation > 1:
        motion_mask = F.max_pool2d(
            raw_motion,
            kernel_size=dilation,
            stride=1,
            padding=dilation // 2,
        )
    else:
        motion_mask = raw_motion
    motion = _masked_mean(squared, motion_mask)

    foreground_mask = (
        real_next.abs() > foreground_threshold
    ).to(prediction.dtype)
    foreground = _masked_mean(squared, foreground_mask)

    target_delta = real_next - real_current
    predicted_delta = prediction - real_current
    delta_squared = (predicted_delta - target_delta).pow(2)
    delta = _masked_mean(delta_squared, motion_mask)

    # Ghost mask: an object/pixel is present now but should be dark next.
    # This directly targets old ball/paddle positions that remain visible.
    departed = (
        (real_current > foreground_threshold)
        & (real_next <= background_threshold)
    ).to(prediction.dtype)
    if dilation > 1:
        departed = F.max_pool2d(
            departed,
            kernel_size=dilation,
            stride=1,
            padding=dilation // 2,
        )

    # Only penalize brightness that exceeds the real next frame by a margin.
    ghost_excess = F.relu(prediction - real_next - ghost_margin).pow(2)
    ghost = _masked_mean(ghost_excess, departed)

    pred_dx, pred_dy = _edge_map(prediction)
    real_dx, real_dy = _edge_map(real_next)
    edge = (
        F.l1_loss(pred_dx, real_dx)
        + F.l1_loss(pred_dy, real_dy)
    ) * 0.5

    total = (
        base
        + motion_weight * motion
        + foreground_weight * foreground
        + delta_weight * delta
        + ghost_weight * ghost
        + edge_weight * edge
    )

    return V4LossParts(
        total=total,
        base=base.detach(),
        motion=motion.detach(),
        foreground=foreground.detach(),
        delta=delta.detach(),
        ghost=ghost.detach(),
        edge=edge.detach(),
        motion_fraction=motion_mask.mean().detach(),
        ghost_fraction=departed.mean().detach(),
    )


def per_sample_drift_error(
    prediction: torch.Tensor,
    real_current: torch.Tensor,
    real_next: torch.Tensor,
    motion_threshold: float = 0.015,
) -> torch.Tensor:
    """Balanced per-sample drift score used by Adaptive correction."""
    base = (prediction - real_next).pow(2).flatten(1).mean(1)
    mask = ((real_next - real_current).abs() > motion_threshold).to(prediction.dtype)
    mask = F.max_pool2d(mask, 3, stride=1, padding=1)
    squared = (prediction - real_next).pow(2)
    motion_sum = (squared * mask).flatten(1).sum(1)
    motion_count = mask.flatten(1).sum(1).clamp_min(1.0)
    motion = motion_sum / motion_count
    return base + motion
