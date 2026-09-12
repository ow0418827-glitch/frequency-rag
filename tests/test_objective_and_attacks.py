from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from frequency_rag.attacks import AttackMethod, run_attack
from frequency_rag.frequency import DCTBasisCache
from frequency_rag.objective import (
    TargetCache,
    joint_objective,
    prepare_targets,
    sequential_image_gradient,
)


def _images() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.linspace(0.1, 0.9, 3 * 5 * 7).reshape(1, 3, 5, 7)
    target = torch.flip(source, dims=(-1,)).mul(0.8).add(0.05)
    return source, target


def _pil_images() -> tuple[Image.Image, Image.Image]:
    source = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3) + 50
    target = np.flip(source, axis=1).copy()
    return Image.fromarray(source, mode="RGB"), Image.fromarray(target, mode="RGB")


def test_target_features_centers_and_text_are_cached(cpu_config, tiny_surrogates) -> None:
    _, target = _images()
    cache = TargetCache()
    first = prepare_targets(
        tiny_surrogates,
        target,
        "target text",
        cpu_config.attack,
        cache=cache,
        target_image_key="target-sha",
    )
    second = prepare_targets(
        tiny_surrogates,
        target,
        "target text",
        cpu_config.attack,
        cache=cache,
        target_image_key="target-sha",
    )
    assert first.image_features[0].global_embedding.data_ptr() == second.image_features[0].global_embedding.data_ptr()
    assert cache.stats()["hits"] == {"image": 3, "centers": 3, "text": 1}
    assert [surrogate.image_calls for surrogate in tiny_surrogates] == [1, 1, 1]
    assert tiny_surrogates[cpu_config.attack.text_model_index].text_calls == 1


def test_cache_key_invalidates_when_text_or_image_changes(cpu_config, tiny_surrogates) -> None:
    _, target = _images()
    cache = TargetCache()
    prepare_targets(
        tiny_surrogates,
        target,
        "first text",
        cpu_config.attack,
        cache=cache,
        target_image_key="image-one",
    )
    prepare_targets(
        tiny_surrogates,
        target,
        "second text",
        cpu_config.attack,
        cache=cache,
        target_image_key="image-two",
    )
    stats = cache.stats()
    assert stats["entries"] == {"image": 6, "centers": 6, "text": 2}


def test_sequential_gradient_matches_joint_graph(cpu_config, tiny_surrogates) -> None:
    source, target = _images()
    targets = prepare_targets(
        tiny_surrogates,
        target,
        "target text",
        cpu_config.attack,
        target_image_key="target",
    )
    weights = torch.tensor([0.2, 0.3, 0.5])

    joint_image = source.clone().requires_grad_(True)
    joint = joint_objective(
        joint_image,
        tiny_surrogates,
        targets,
        weights,
        cpu_config.attack,
        use_cached_target_centers=True,
    )
    joint_gradient = torch.autograd.grad(joint.loss, joint_image)[0]

    sequential_image = source.clone().requires_grad_(True)
    sequential_gradient, sequential = sequential_image_gradient(
        sequential_image,
        tiny_surrogates,
        targets,
        weights,
        cpu_config.attack,
    )
    torch.testing.assert_close(sequential.loss, joint.loss.detach(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        sequential.global_similarities,
        joint.global_similarities.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    relative = torch.linalg.vector_norm(sequential_gradient - joint_gradient) / torch.linalg.vector_norm(joint_gradient)
    assert relative.item() <= 1e-4


@pytest.mark.parametrize(
    "method",
    [AttackMethod.LEGACY_PIXEL, AttackMethod.FAIR_PIXEL, AttackMethod.FREQUENCY],
)
def test_attack_methods_save_budget_compliant_png_and_true_final_metrics(
    tmp_path: Path,
    cpu_config,
    tiny_surrogates,
    method: AttackMethod,
) -> None:
    source, target = _pil_images()
    output = tmp_path / method.value
    result = run_attack(
        source,
        target,
        "a target description",
        cpu_config,
        method=method,
        steps=3,
        output_directory=output,
        surrogates=tiny_surrogates,
        target_cache=TargetCache(),
        basis_cache=DCTBasisCache(),
        target_image_key="target-sha",
    )
    metadata = result.metadata
    assert metadata["status"] == "success"
    assert metadata["attack"]["steps_completed"] == 3
    assert metadata["history_semantics"]["final_objective_is_recomputed_after_last_update"]
    assert metadata["decoded_image_audit"]["within_budget"]
    assert metadata["decoded_image_audit"]["max_absolute_pixel_difference_levels"] <= 16
    assert (output / "adversarial.png").is_file()
    assert (output / "attack_metrics.json").is_file()
    assert len(metadata["checkpoints"]) == 4
    assert metadata["primary_evaluator_used_during_attack"] is False
    saved = np.asarray(Image.open(output / "adversarial.png").convert("RGB"))
    assert saved.shape == np.asarray(source).shape


def test_frequency_attack_uses_compact_parameter_count(tmp_path, cpu_config, tiny_surrogates) -> None:
    source, target = _pil_images()
    projection_config = replace(
        cpu_config,
        attack=replace(
            cpu_config.attack,
            epsilon=0.01,
            spatial_step_size=0.1,
        ),
    )
    result = run_attack(
        source,
        target,
        "target",
        projection_config,
        method="frequency",
        steps=1,
        axis_ratio=0.25,
        output_directory=tmp_path / "frequency",
        surrogates=tiny_surrogates,
    )
    frequency = result.metadata["frequency"]
    assert frequency["coefficient_count"] == 3 * 2 * 2
    assert frequency["spatial_parameter_count"] == 3 * 5 * 7
    assert frequency["coefficient_count"] < frequency["spatial_parameter_count"]
    assert frequency["radial_scale_summary"]["count"] == 1
    assert frequency["projection"] == (
        "spatial_clip_low_frequency_reproject_with_radial_fallback"
    )
    assert frequency["maximum_reprojection_iterations"] == 1
    assert frequency["reprojection_summary"]["count"] == 1
    assert frequency["reprojection_summary"]["steps_with_spatial_clip"] == 1
    assert result.metadata["history"][0]["projection_iterations"] == 1
    assert result.metadata["decoded_image_audit"]["within_budget"]


def test_progressive_frequency_embeds_old_coefficients(tmp_path, cpu_config, tiny_surrogates) -> None:
    source, target = _pil_images()
    progressive = replace(cpu_config.frequency.progressive, enabled=True)
    config = replace(cpu_config, frequency=replace(cpu_config.frequency, progressive=progressive))
    result = run_attack(
        source,
        target,
        "target",
        config,
        method="progressive_frequency",
        steps=4,
        output_directory=tmp_path / "progressive",
        surrogates=tiny_surrogates,
    )
    frequency = result.metadata["frequency"]
    assert frequency["progressive_schedule_basis"] == "step_fraction_fallback"
    assert frequency["axis_ratio_final"] == 0.5
    assert len(frequency["transitions"]) == 2


def test_progressive_frequency_requires_explicit_enablement(
    tmp_path, cpu_config, tiny_surrogates
) -> None:
    source, target = _pil_images()
    with pytest.raises(ValueError, match="渐进扩频在配置中未启用"):
        run_attack(
            source,
            target,
            "target",
            cpu_config,
            method="progressive_frequency",
            steps=1,
            output_directory=tmp_path / "progressive-disabled",
            surrogates=tiny_surrogates,
        )


def test_progressive_frequency_rejects_conflicting_axis_ratio(
    tmp_path, cpu_config, tiny_surrogates
) -> None:
    source, target = _pil_images()
    progressive = replace(cpu_config.frequency.progressive, enabled=True)
    config = replace(cpu_config, frequency=replace(cpu_config.frequency, progressive=progressive))
    with pytest.raises(ValueError, match="不能同时传入单一 axis_ratio"):
        run_attack(
            source,
            target,
            "target",
            config,
            method="progressive_frequency",
            steps=1,
            axis_ratio=0.5,
            output_directory=tmp_path / "progressive-conflict",
            surrogates=tiny_surrogates,
        )


def test_pixel_method_rejects_unused_axis_ratio(
    tmp_path, cpu_config, tiny_surrogates
) -> None:
    source, target = _pil_images()
    with pytest.raises(ValueError, match="像素方法和零步对照不使用 axis_ratio"):
        run_attack(
            source,
            target,
            "target",
            cpu_config,
            method="fair_pixel",
            steps=1,
            axis_ratio=0.25,
            output_directory=tmp_path / "pixel-axis-ratio",
            surrogates=tiny_surrogates,
        )


def test_short_time_budget_never_commits_over_budget_step(tmp_path, cpu_config, tiny_surrogates) -> None:
    source, target = _pil_images()
    result = run_attack(
        source,
        target,
        "target",
        cpu_config,
        method="fair_pixel",
        steps=100,
        time_budget_seconds=1e-9,
        output_directory=tmp_path / "budget",
        surrogates=tiny_surrogates,
    )
    assert result.metadata["attack"]["steps_completed"] == 0
    assert result.metadata["attack"]["stop_reason"] == "time_budget_exhausted"
    assert result.metadata["decoded_image_audit"]["max_absolute_pixel_difference_levels"] == 0


def test_zero_baseline_preserves_decoded_image(tmp_path, cpu_config, tiny_surrogates) -> None:
    source, target = _pil_images()
    result = run_attack(
        source,
        target,
        "target",
        cpu_config,
        method="zero",
        steps=3,
        output_directory=tmp_path / "zero",
        surrogates=tiny_surrogates,
    )
    assert result.metadata["attack"]["steps_completed"] == 0
    assert result.metadata["decoded_image_audit"]["max_absolute_pixel_difference_levels"] == 0
    np.testing.assert_array_equal(np.asarray(source), np.asarray(result.image))
