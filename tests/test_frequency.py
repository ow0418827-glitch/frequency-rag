from __future__ import annotations

import math

import pytest
import torch

from frequency_rag.frequency import (
    DCTBasisCache,
    ProgressiveSchedule,
    analyse,
    axis_frequency_count,
    expand_coefficients,
    orthonormal_dct_basis,
    out_of_band_energy_ratio,
    project_and_reproject,
    synthesize,
    update_coefficients,
)


@pytest.mark.parametrize("length", [1, 2, 5, 8, 11])
def test_full_basis_is_orthonormal_in_double_precision(length: int) -> None:
    basis = orthonormal_dct_basis(length, dtype=torch.float64)
    identity = torch.eye(length, dtype=torch.float64)
    assert torch.max(torch.abs(basis.T @ basis - identity)).item() <= 1e-12


@pytest.mark.parametrize("shape", [(5, 7), (8, 8), (9, 4)])
def test_full_transform_reconstruction_and_energy(shape: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(42)
    image = torch.randn(1, 3, *shape, generator=generator, dtype=torch.float64)
    height_basis = orthonormal_dct_basis(shape[0], dtype=torch.float64)
    width_basis = orthonormal_dct_basis(shape[1], dtype=torch.float64)
    coefficients = analyse(image, height_basis, width_basis)
    reconstructed = synthesize(coefficients, height_basis, width_basis)
    assert torch.max(torch.abs(image - reconstructed)).item() <= 1e-10
    assert abs(image.square().sum().item() - coefficients.square().sum().item()) <= 1e-10


def test_compact_synthesis_matches_zero_padded_full_synthesis() -> None:
    height, width, kh, kw = 7, 9, 3, 4
    coefficients = torch.randn(1, 3, kh, kw, dtype=torch.float64)
    compact_h = orthonormal_dct_basis(height, kh, dtype=torch.float64)
    compact_w = orthonormal_dct_basis(width, kw, dtype=torch.float64)
    compact = synthesize(coefficients, compact_h, compact_w)
    full_coefficients = torch.zeros(1, 3, height, width, dtype=torch.float64)
    full_coefficients[..., :kh, :kw] = coefficients
    full = synthesize(
        full_coefficients,
        orthonormal_dct_basis(height, dtype=torch.float64),
        orthonormal_dct_basis(width, dtype=torch.float64),
    )
    torch.testing.assert_close(compact, full, atol=1e-12, rtol=1e-12)


def test_compact_coefficient_gradient_matches_finite_difference() -> None:
    height_basis = orthonormal_dct_basis(5, 3, dtype=torch.float64)
    width_basis = orthonormal_dct_basis(7, 4, dtype=torch.float64)
    coefficients = torch.randn(1, 3, 3, 4, dtype=torch.float64, requires_grad=True) * 0.01
    weights = torch.randn(1, 3, 5, 7, dtype=torch.float64)
    value = torch.sin(synthesize(coefficients, height_basis, width_basis) * weights).sum()
    gradient = torch.autograd.grad(value, coefficients)[0]
    index = (0, 1, 2, 3)
    epsilon = 1e-6
    with torch.no_grad():
        plus = coefficients.detach().clone()
        minus = coefficients.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = torch.sin(synthesize(plus, height_basis, width_basis) * weights).sum()
        minus_value = torch.sin(synthesize(minus, height_basis, width_basis) * weights).sum()
    finite_difference = (plus_value - minus_value) / (2 * epsilon)
    relative = abs(gradient[index].item() - finite_difference.item()) / max(
        abs(finite_difference.item()), 1e-12
    )
    assert relative <= 1e-6


def test_frequency_update_calibrates_spatial_step_and_budget() -> None:
    height_basis = orthonormal_dct_basis(9, 3)
    width_basis = orthonormal_dct_basis(11, 4)
    coefficients = torch.zeros(1, 3, 3, 4)
    gradient = torch.linspace(-1, 1, coefficients.numel()).reshape_as(coefficients)
    first = update_coefficients(
        coefficients,
        gradient,
        height_basis,
        width_basis,
        spatial_step_size=0.5 / 255,
        epsilon=16 / 255,
    )
    assert first.zero_gradient is False
    assert first.resulting_linf == pytest.approx(0.5 / 255, rel=1e-5, abs=1e-7)
    assert first.projection_iterations == 0
    assert first.initial_clipped_fraction == 0.0
    assert first.radial_scale == 1.0
    current = first.coefficients
    for _ in range(100):
        current = update_coefficients(
            current,
            gradient,
            height_basis,
            width_basis,
            spatial_step_size=0.5 / 255,
            epsilon=16 / 255,
        ).coefficients
    assert synthesize(current, height_basis, width_basis).abs().max().item() <= 16 / 255 + 1e-6


def test_clip_reproject_rechecks_overshoot_and_uses_radial_fallback() -> None:
    height_basis = orthonormal_dct_basis(3, 2, dtype=torch.float64)
    width_basis = orthonormal_dct_basis(1, 1, dtype=torch.float64)
    spatial = torch.tensor([17.0, 9.0, 1.0], dtype=torch.float64).reshape(1, 1, 3, 1)
    coefficients = analyse(spatial, height_basis, width_basis)

    result = project_and_reproject(
        coefficients,
        height_basis,
        width_basis,
        epsilon=16.0,
        maximum_iterations=1,
        convergence_tolerance=1e-12,
    )
    projected_spatial = synthesize(result.coefficients, height_basis, width_basis)
    direct_scaled = spatial * result.direct_radial_scale

    assert result.initial_clipped_fraction == pytest.approx(1 / 3)
    assert result.projection_iterations == 1
    assert result.post_reprojection_linf > 16.0
    assert result.converged_before_fallback is False
    assert result.radial_scale < 1.0
    assert result.radial_scale > result.direct_radial_scale
    assert result.resulting_linf <= 16.0
    assert projected_spatial.abs().mean() > direct_scaled.abs().mean()
    assert out_of_band_energy_ratio(
        projected_spatial, height_basis, width_basis
    ) <= 1e-12


def test_clip_reproject_keeps_constant_component_disabled() -> None:
    height_basis = orthonormal_dct_basis(5, 3, dtype=torch.float64)
    width_basis = orthonormal_dct_basis(5, 3, dtype=torch.float64)
    coefficients = torch.ones(1, 3, 3, 3, dtype=torch.float64) * 10

    result = project_and_reproject(
        coefficients,
        height_basis,
        width_basis,
        epsilon=0.2,
        maximum_iterations=2,
        keep_constant_component=False,
    )

    assert torch.count_nonzero(result.coefficients[..., 0, 0]) == 0
    assert result.resulting_linf <= 0.2


def test_zero_gradient_is_explicit_and_finite() -> None:
    height_basis = orthonormal_dct_basis(5, 2)
    width_basis = orthonormal_dct_basis(7, 3)
    coefficients = torch.randn(1, 3, 2, 3) * 0.01
    result = update_coefficients(
        coefficients,
        torch.zeros_like(coefficients),
        height_basis,
        width_basis,
        spatial_step_size=0.1,
        epsilon=0.2,
    )
    assert result.zero_gradient
    assert torch.isfinite(result.coefficients).all()
    torch.testing.assert_close(result.coefficients, coefficients)


def test_constant_component_can_be_removed() -> None:
    height_basis = orthonormal_dct_basis(4, 2)
    width_basis = orthonormal_dct_basis(4, 2)
    coefficients = torch.zeros(1, 3, 2, 2)
    gradient = torch.zeros_like(coefficients)
    gradient[..., 0, 0] = 1
    result = update_coefficients(
        coefficients,
        gradient,
        height_basis,
        width_basis,
        spatial_step_size=0.1,
        epsilon=0.2,
        keep_constant_component=False,
    )
    assert result.zero_gradient
    assert torch.count_nonzero(result.coefficients) == 0


def test_raw_compact_perturbation_has_negligible_out_of_band_energy() -> None:
    height_basis = orthonormal_dct_basis(9, 3, dtype=torch.float64)
    width_basis = orthonormal_dct_basis(7, 2, dtype=torch.float64)
    coefficients = torch.randn(1, 3, 3, 2, dtype=torch.float64)
    spatial = synthesize(coefficients, height_basis, width_basis)
    assert out_of_band_energy_ratio(spatial, height_basis, width_basis) <= 1e-12


def test_expand_coefficients_preserves_old_block() -> None:
    coefficients = torch.randn(1, 3, 2, 3)
    expanded = expand_coefficients(coefficients, 4, 5)
    torch.testing.assert_close(expanded[..., :2, :3], coefficients)
    assert torch.count_nonzero(expanded[..., 2:, :]) == 0
    assert torch.count_nonzero(expanded[..., :2, 3:]) == 0


def test_axis_count_uses_ceiling() -> None:
    assert axis_frequency_count(7, 0.25) == 2
    assert axis_frequency_count(8, 0.25) == 2
    assert axis_frequency_count(1, 0.125) == 1


def test_progressive_schedule_boundaries() -> None:
    schedule = ProgressiveSchedule((0.0, 0.3, 0.7), (0.125, 0.25, 0.5))
    assert schedule.ratio_at(0.0) == 0.125
    assert schedule.ratio_at(math.nextafter(0.3, 0.0)) == 0.125
    assert schedule.ratio_at(0.3) == 0.25
    assert schedule.ratio_at(0.7) == 0.5


def test_basis_cache_keys_include_shape_dtype_and_records_hits() -> None:
    cache = DCTBasisCache(maximum_entries=2)
    first = cache.get(7, 3, device="cpu", dtype=torch.float32)
    second = cache.get(7, 3, device="cpu", dtype=torch.float32)
    assert first.data_ptr() == second.data_ptr()
    cache.get(7, 3, device="cpu", dtype=torch.float64)
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["entries"] == 2
