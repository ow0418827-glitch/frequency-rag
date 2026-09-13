"""原像素扰动更新：梯度符号步长、像素范围和扰动预算投影。"""
from __future__ import annotations
from typing import Any
import torch

def pixel_update(
    delta: torch.Tensor,
    gradient: torch.Tensor,
    source: torch.Tensor,
    *,
    step_size: float,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("像素扰动梯度包含非有限值。")
    with torch.no_grad():
        zero_gradient = not bool(torch.count_nonzero(gradient).detach().cpu())
        proposed = delta + float(step_size) * gradient.sign()
        proposed = proposed.clamp(-float(epsilon), float(epsilon))
        updated = ((source + proposed).clamp(0, 1) - source).detach()
        linf = float(updated.abs().amax().detach().cpu())
    return updated, {
        "zero_gradient": zero_gradient,
        "resulting_linf": linf,
        "radial_scale": None,
        "direction_linf": 1.0 if not zero_gradient else 0.0,
        "proposed_linf": float(proposed.abs().amax().detach().cpu()),
        "projection_iterations": None,
        "initial_clipped_fraction": None,
        "post_reprojection_linf": None,
        "converged_before_fallback": None,
        "direct_radial_scale": None,
    }
