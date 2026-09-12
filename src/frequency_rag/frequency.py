from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any

import torch


def axis_frequency_count(length: int, ratio: float) -> int:
    if length <= 0:
        raise ValueError("空间尺寸必须为正整数。")
    if not 0 < ratio <= 1:
        raise ValueError("频率方向比例必须位于 (0, 1]。")
    return min(length, max(1, math.ceil(length * float(ratio))))


def orthonormal_dct_basis(
    length: int,
    frequencies: int | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """返回形状为 [空间位置, 频率] 的正交二型离散余弦基底。"""
    if length <= 0:
        raise ValueError("基底长度必须为正整数。")
    count = length if frequencies is None else int(frequencies)
    if not 1 <= count <= length:
        raise ValueError("频率数量必须位于 [1, 空间长度]。")
    if not (dtype.is_floating_point or dtype.is_complex):
        raise TypeError("离散余弦基底需要浮点数据类型。")
    real_dtype = torch.float64 if dtype == torch.complex128 else (
        torch.float32 if dtype == torch.complex64 else dtype
    )
    positions = torch.arange(length, device=device, dtype=real_dtype).unsqueeze(1)
    indices = torch.arange(count, device=device, dtype=real_dtype).unsqueeze(0)
    basis = torch.cos(math.pi * (2 * positions + 1) * indices / (2 * length))
    scale = torch.full((count,), math.sqrt(2.0 / length), device=device, dtype=real_dtype)
    scale[0] = 1.0 / math.sqrt(length)
    result = basis * scale.unsqueeze(0)
    return result.to(dtype=dtype).contiguous()


class DCTBasisCache:
    """按尺寸、设备和精度复用紧凑基底；最旧条目会被自动淘汰。"""

    def __init__(self, maximum_entries: int = 24) -> None:
        if maximum_entries <= 0:
            raise ValueError("基底缓存容量必须为正整数。")
        self.maximum_entries = int(maximum_entries)
        self._values: OrderedDict[tuple[int, int, str, torch.dtype], torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(
        self,
        length: int,
        frequencies: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        resolved_device = str(torch.device(device))
        key = (int(length), int(frequencies), resolved_device, dtype)
        if key in self._values:
            self.hits += 1
            self._values.move_to_end(key)
            return self._values[key]
        self.misses += 1
        value = orthonormal_dct_basis(
            length, frequencies, device=resolved_device, dtype=dtype
        )
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.maximum_entries:
            self._values.popitem(last=False)
        return value

    def clear(self) -> None:
        self._values.clear()

    def stats(self) -> dict[str, int]:
        bytes_used = sum(value.numel() * value.element_size() for value in self._values.values())
        return {
            "entries": len(self._values),
            "hits": self.hits,
            "misses": self.misses,
            "bytes": bytes_used,
        }


def synthesize(coefficients: torch.Tensor, height_basis: torch.Tensor, width_basis: torch.Tensor) -> torch.Tensor:
    """由紧凑系数合成 [批, 通道, 高, 宽] 空间扰动。"""
    if coefficients.ndim != 4:
        raise ValueError("频率系数必须具有 [批, 通道, 高频数, 宽频数] 形状。")
    if height_basis.ndim != 2 or width_basis.ndim != 2:
        raise ValueError("方向基底必须是二维矩阵。")
    if coefficients.shape[-2:] != (height_basis.shape[1], width_basis.shape[1]):
        raise ValueError("频率系数尺寸与紧凑基底不一致。")
    return torch.einsum("hk,bckl,wl->bchw", height_basis, coefficients, width_basis)


def analyse(spatial: torch.Tensor, height_basis: torch.Tensor, width_basis: torch.Tensor) -> torch.Tensor:
    """把空间张量投影到给定的紧凑正交基底。"""
    if spatial.ndim != 4:
        raise ValueError("空间张量必须具有 [批, 通道, 高, 宽] 形状。")
    if spatial.shape[-2:] != (height_basis.shape[0], width_basis.shape[0]):
        raise ValueError("空间张量尺寸与基底不一致。")
    return torch.einsum("hk,bchw,wl->bckl", height_basis, spatial, width_basis)


def expand_coefficients(
    coefficients: torch.Tensor,
    height_frequencies: int,
    width_frequencies: int,
) -> torch.Tensor:
    old_height, old_width = coefficients.shape[-2:]
    if height_frequencies < old_height or width_frequencies < old_width:
        raise ValueError("渐进扩频不能缩小已有系数阵列。")
    if (height_frequencies, width_frequencies) == (old_height, old_width):
        return coefficients
    expanded = coefficients.new_zeros(
        *coefficients.shape[:-2], height_frequencies, width_frequencies
    )
    expanded[..., :old_height, :old_width] = coefficients
    return expanded


@dataclass(frozen=True)
class FrequencyUpdate:
    coefficients: torch.Tensor
    zero_gradient: bool
    direction_linf: float
    proposed_linf: float
    projection_iterations: int
    initial_clipped_fraction: float
    post_reprojection_linf: float
    converged_before_fallback: bool
    direct_radial_scale: float
    radial_scale: float
    resulting_linf: float


@dataclass(frozen=True)
class FeasibilityProjection:
    coefficients: torch.Tensor
    initial_linf: float
    projection_iterations: int
    initial_clipped_fraction: float
    post_reprojection_linf: float
    converged_before_fallback: bool
    direct_radial_scale: float
    radial_scale: float
    resulting_linf: float


def project_and_reproject(
    coefficients: torch.Tensor,
    height_basis: torch.Tensor,
    width_basis: torch.Tensor,
    *,
    epsilon: float,
    maximum_iterations: int = 1,
    convergence_tolerance: float = 1e-7,
    keep_constant_component: bool = True,
) -> FeasibilityProjection:
    """在空间预算盒与低频子空间之间交替投影，并以整体缩放严格兜底。"""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("扰动预算必须为有限正数。")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or not 1 <= maximum_iterations <= 16
    ):
        raise ValueError("空间截断与低频重投影的最大轮数必须位于 [1, 16]。")
    if not math.isfinite(convergence_tolerance) or convergence_tolerance < 0:
        raise ValueError("重投影收敛容差必须为有限非负数。")
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError("待投影频率系数包含非有限值。")

    projected = coefficients.detach().clone()
    if not keep_constant_component:
        projected[..., 0, 0] = 0
    spatial = synthesize(projected, height_basis, width_basis)
    initial_linf = float(spatial.abs().amax().detach().cpu())
    initial_clipped_fraction = float(
        (spatial.abs() > float(epsilon)).to(dtype=torch.float32).mean().detach().cpu()
    )
    direct_radial_scale = min(
        1.0,
        float(epsilon) / max(initial_linf, torch.finfo(spatial.dtype).tiny),
    )
    projection_iterations = 0

    # 超限后固定执行有限轮，避免每一轮读取设备标量造成同步开销。达到可行域后，
    # 截断与同一子空间正交投影均为幂等操作，后续轮不会有意改变结果。
    if initial_linf > float(epsilon) + convergence_tolerance:
        for _ in range(maximum_iterations):
            spatial_clamped = spatial.clamp(min=-float(epsilon), max=float(epsilon))
            projected = analyse(spatial_clamped, height_basis, width_basis)
            if not keep_constant_component:
                projected[..., 0, 0] = 0
            spatial = synthesize(projected, height_basis, width_basis)
            projection_iterations += 1

    post_reprojection_linf = float(spatial.abs().amax().detach().cpu())
    converged_before_fallback = post_reprojection_linf <= float(epsilon) + convergence_tolerance
    radial_scale = min(
        1.0,
        float(epsilon) / max(post_reprojection_linf, torch.finfo(spatial.dtype).tiny),
    )
    if radial_scale < 1.0:
        projected = projected * radial_scale
        spatial = synthesize(projected, height_basis, width_basis)
    resulting_linf = float(spatial.abs().amax().detach().cpu())

    # 浮点乘法和再次合成可能产生极小的向上舍入；留出机器精度裕量校正，
    # 使返回的浮点扰动本身也不高于名义预算，而不只是在容差内通过。
    safety_factor = max(0.0, 1.0 - 4.0 * torch.finfo(spatial.dtype).eps)
    for _ in range(2):
        if resulting_linf <= float(epsilon):
            break
        correction = float(epsilon) * safety_factor / resulting_linf
        projected = projected * correction
        radial_scale *= correction
        spatial = synthesize(projected, height_basis, width_basis)
        resulting_linf = float(spatial.abs().amax().detach().cpu())

    return FeasibilityProjection(
        coefficients=projected.detach(),
        initial_linf=initial_linf,
        projection_iterations=projection_iterations,
        initial_clipped_fraction=initial_clipped_fraction,
        post_reprojection_linf=post_reprojection_linf,
        converged_before_fallback=converged_before_fallback,
        direct_radial_scale=direct_radial_scale,
        radial_scale=radial_scale,
        resulting_linf=resulting_linf,
    )


def update_coefficients(
    coefficients: torch.Tensor,
    gradient: torch.Tensor,
    height_basis: torch.Tensor,
    width_basis: torch.Tensor,
    *,
    spatial_step_size: float,
    epsilon: float,
    keep_constant_component: bool = True,
    maximum_reprojection_iterations: int = 1,
    reprojection_tolerance: float = 1e-7,
    zero_threshold: float = 1e-12,
) -> FrequencyUpdate:
    """按系数梯度符号更新，再经空间截断、低频重投影与缩放兜底。"""
    if coefficients.shape != gradient.shape:
        raise ValueError("系数与梯度形状不一致。")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("频率系数梯度包含非有限值。")
    with torch.no_grad():
        direction = gradient.sign()
        if not keep_constant_component:
            direction[..., 0, 0] = 0
        spatial_direction = synthesize(direction, height_basis, width_basis)
        direction_linf_tensor = spatial_direction.abs().amax()
        direction_linf = float(direction_linf_tensor.detach().cpu())
        if direction_linf <= zero_threshold:
            current = synthesize(coefficients, height_basis, width_basis)
            current_linf = float(current.abs().amax().detach().cpu())
            return FrequencyUpdate(
                coefficients=coefficients.detach().clone(),
                zero_gradient=True,
                direction_linf=direction_linf,
                proposed_linf=current_linf,
                projection_iterations=0,
                initial_clipped_fraction=0.0,
                post_reprojection_linf=current_linf,
                converged_before_fallback=current_linf <= float(epsilon) + reprojection_tolerance,
                direct_radial_scale=1.0,
                radial_scale=1.0,
                resulting_linf=current_linf,
            )
        calibrated_step = float(spatial_step_size) / max(direction_linf, zero_threshold)
        proposed = coefficients + calibrated_step * direction
        if not keep_constant_component:
            proposed[..., 0, 0] = 0
        projection = project_and_reproject(
            proposed,
            height_basis,
            width_basis,
            epsilon=epsilon,
            maximum_iterations=maximum_reprojection_iterations,
            convergence_tolerance=reprojection_tolerance,
            keep_constant_component=keep_constant_component,
        )
        proposed_linf = projection.initial_linf
    return FrequencyUpdate(
        coefficients=projection.coefficients,
        zero_gradient=False,
        direction_linf=direction_linf,
        proposed_linf=proposed_linf,
        projection_iterations=projection.projection_iterations,
        initial_clipped_fraction=projection.initial_clipped_fraction,
        post_reprojection_linf=projection.post_reprojection_linf,
        converged_before_fallback=projection.converged_before_fallback,
        direct_radial_scale=projection.direct_radial_scale,
        radial_scale=projection.radial_scale,
        resulting_linf=projection.resulting_linf,
    )


def out_of_band_energy_ratio(
    spatial: torch.Tensor,
    height_basis: torch.Tensor,
    width_basis: torch.Tensor,
    *,
    stabilizer: float = 1e-12,
) -> float:
    """利用正交投影计算给定低频矩形之外的能量占比。"""
    projected = analyse(spatial, height_basis, width_basis)
    total_energy = spatial.double().square().sum()
    in_band_energy = projected.double().square().sum()
    ratio = (total_energy - in_band_energy).clamp_min(0) / (total_energy + stabilizer)
    return float(ratio.clamp(0, 1).detach().cpu())


def perturbation_diagnostics(
    source: torch.Tensor,
    raw_perturbation: torch.Tensor,
    final_float_image: torch.Tensor,
    decoded_image: torch.Tensor,
    height_basis: torch.Tensor,
    width_basis: torch.Tensor,
) -> dict[str, Any]:
    if source.shape != raw_perturbation.shape or source.shape != final_float_image.shape:
        raise ValueError("源图、原始扰动与最终浮点图片形状必须一致。")
    if decoded_image.shape != source.shape:
        raise ValueError("重新读取图片的形状与源图不一致。")
    actual_float = final_float_image - source
    decoded = decoded_image - source
    clipped = ((source + raw_perturbation) < 0) | ((source + raw_perturbation) > 1)

    def stats(value: torch.Tensor) -> dict[str, float]:
        detached = value.detach()
        return {
            "linf": float(detached.abs().amax().cpu()),
            "mean_absolute": float(detached.abs().mean().cpu()),
            "mean_squared": float(detached.square().mean().cpu()),
            "out_of_band_energy_ratio": out_of_band_energy_ratio(
                detached, height_basis, width_basis
            ),
        }

    return {
        "raw_synthesized": stats(raw_perturbation),
        "after_range_clip": stats(actual_float),
        "after_png_reload": stats(decoded),
        "range_clipped_element_fraction": float(clipped.float().mean().detach().cpu()),
    }


@dataclass(frozen=True)
class ProgressiveSchedule:
    time_fractions: tuple[float, ...]
    axis_ratios: tuple[float, ...]

    def ratio_at(self, progress_fraction: float) -> float:
        progress = max(0.0, float(progress_fraction))
        selected = self.axis_ratios[0]
        for boundary, ratio in zip(self.time_fractions, self.axis_ratios, strict=True):
            if progress < boundary:
                break
            selected = ratio
        return selected
