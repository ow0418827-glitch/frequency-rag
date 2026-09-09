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
    radial_scale: float
    resulting_linf: float


def update_coefficients(
    coefficients: torch.Tensor,
    gradient: torch.Tensor,
    height_basis: torch.Tensor,
    width_basis: torch.Tensor,
    *,
    spatial_step_size: float,
    epsilon: float,
    keep_constant_component: bool = True,
    zero_threshold: float = 1e-12,
) -> FrequencyUpdate:
    """按系数梯度符号更新，并用空间无穷范数校准和整体缩放。"""
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
                radial_scale=1.0,
                resulting_linf=current_linf,
            )
        calibrated_step = float(spatial_step_size) / max(direction_linf, zero_threshold)
        proposed = coefficients + calibrated_step * direction
        if not keep_constant_component:
            proposed[..., 0, 0] = 0
        proposed_spatial = synthesize(proposed, height_basis, width_basis)
        proposed_linf = float(proposed_spatial.abs().amax().detach().cpu())
        radial_scale = min(1.0, float(epsilon) / max(proposed_linf, zero_threshold))
        updated = (proposed * radial_scale).detach()
        resulting = synthesize(updated, height_basis, width_basis)
        resulting_linf = float(resulting.abs().amax().detach().cpu())
    return FrequencyUpdate(
        coefficients=updated,
        zero_gradient=False,
        direction_linf=direction_linf,
        proposed_linf=proposed_linf,
        radial_scale=radial_scale,
        resulting_linf=resulting_linf,
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
