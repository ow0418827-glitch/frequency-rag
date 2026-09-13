from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from PIL import Image
import torch

from frequency_rag.pixel_attack.update import pixel_update as _pixel_update
from frequency_rag.common.config import ProjectConfig
from frequency_rag.frequency_attack.dct import (
    DCTBasisCache,
    ProgressiveSchedule,
    axis_frequency_count,
    expand_coefficients,
    perturbation_diagnostics,
    synthesize,
    update_coefficients,
)
from frequency_rag.common.io import (
    decoded_image_audit,
    load_image,
    pil_to_tensor,
    save_json,
    save_tensor_png,
    sha256_file,
    tensor_to_pil,
)
from frequency_rag.attack_core.surrogates import load_surrogates, resolve_device
from frequency_rag.attack_core.objective import (
    TargetCache,
    evaluate_objective,
    joint_objective,
    prepare_targets,
    sequential_image_gradient,
)
from frequency_rag.common.profiling import DeviceMemoryMonitor, PhaseClock, synchronize


class AttackMethod(str, Enum):
    ZERO = "zero"
    LEGACY_PIXEL = "legacy_pixel"
    FAIR_PIXEL = "fair_pixel"
    FREQUENCY = "frequency"
    PROGRESSIVE_FREQUENCY = "progressive_frequency"

    @classmethod
    def parse(cls, value: str | "AttackMethod") -> "AttackMethod":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = "、".join(item.value for item in cls)
            raise ValueError(f"未知攻击方法 {value!r}；可选值为：{choices}。") from exc


@dataclass(frozen=True)
class AttackResult:
    image: Image.Image
    tensor: torch.Tensor
    decoded_tensor: torch.Tensor
    metadata: dict[str, Any]


def _cache_stats_copy(cache: TargetCache) -> dict[str, Any]:
    return json.loads(json.dumps(cache.stats()))


def _weights_list(weights: torch.Tensor) -> list[float]:
    return [float(value) for value in weights.detach().cpu().tolist()]


def _current_frequency_state(
    source: torch.Tensor,
    coefficients: torch.Tensor,
    height_basis: torch.Tensor,
    width_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = synthesize(coefficients, height_basis, width_basis)
    return raw, (source + raw).clamp(0, 1)


def _checkpoint_label(fraction: float) -> str:
    return f"budget_{int(round(fraction * 1000)):04d}"


def run_attack(
    source_image: str | Path | Image.Image,
    target_image: str | Path | Image.Image,
    target_text: str | None,
    project_config: ProjectConfig,
    *,
    method: str | AttackMethod = AttackMethod.FREQUENCY,
    steps: int | None = None,
    axis_ratio: float | None = None,
    time_budget_seconds: float | None = None,
    output_directory: str | Path | None = None,
    surrogates: Sequence[Any] | None = None,
    target_cache: TargetCache | None = None,
    basis_cache: DCTBasisCache | None = None,
    target_image_key: str | None = None,
    checkpoint_fractions: Sequence[float] | None = None,
) -> AttackResult:
    """运行单个冻结图像对；主评估器不在本函数中加载或查询。"""
    project_config.validate()
    selected_method = AttackMethod.parse(method)
    frequency_method = selected_method in {
        AttackMethod.FREQUENCY,
        AttackMethod.PROGRESSIVE_FREQUENCY,
    }
    if axis_ratio is not None and not frequency_method:
        raise ValueError("像素方法和零步对照不使用 axis_ratio，不能传入该参数。")
    if axis_ratio is not None and (
        not math.isfinite(axis_ratio) or not 0 < axis_ratio <= 1
    ):
        raise ValueError("频率方向比例必须是位于 (0, 1] 的有限数值。")
    if selected_method is AttackMethod.PROGRESSIVE_FREQUENCY:
        if not project_config.frequency.progressive.enabled:
            raise ValueError(
                "渐进扩频在配置中未启用；请先将 frequency.progressive.enabled 设为 true。"
            )
        if axis_ratio is not None:
            raise ValueError(
                "渐进扩频由预先配置的频带日程控制，不能同时传入单一 axis_ratio。"
            )
    requested_steps = project_config.attack.reference_steps if steps is None else int(steps)
    if requested_steps < 0:
        raise ValueError("攻击步数不能为负数。")
    if time_budget_seconds is not None and (
        not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0
    ):
        raise ValueError("时间预算必须为有限正数。")
    fractions = tuple(
        float(value)
        for value in (
            checkpoint_fractions
            if checkpoint_fractions is not None
            else project_config.experiment.checkpoints_budget_fractions
        )
    )
    if any(not 0 < value <= 1 for value in fractions) or tuple(sorted(set(fractions))) != fractions:
        raise ValueError("检查点比例必须严格递增并位于 (0, 1]。")

    attack_config = replace(project_config.attack, reference_steps=requested_steps)
    device = resolve_device(
        project_config.device,
        allow_fallback=project_config.runtime.allow_device_fallback,
    )
    torch.manual_seed(project_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(project_config.seed)

    model_loading_seconds = 0.0
    if surrogates is None:
        loading_clock = PhaseClock(device)
        loading_clock.start()
        surrogates = load_surrogates(
            project_config.surrogates,
            device=str(device),
            precision=project_config.attack_precision,
            allow_downloads=project_config.runtime.allow_downloads,
            allow_device_fallback=project_config.runtime.allow_device_fallback,
            require_true_local_tokens=project_config.runtime.require_true_local_tokens,
            hash_weights=project_config.runtime.hash_model_weights,
        )
        model_loading_seconds = loading_clock.stop()
    if len(surrogates) != 3:
        raise ValueError("兼容主实验需要恰好三个代理模型。")
    for surrogate in surrogates:
        surrogate_device = getattr(surrogate, "device", device)
        if torch.device(surrogate_device) != device:
            raise ValueError("所有代理模型与攻击张量必须位于同一设备。")

    shared_target_cache = target_cache or TargetCache()
    shared_basis_cache = basis_cache or DCTBasisCache()
    cache_before = _cache_stats_copy(shared_target_cache)
    memory_monitor = DeviceMemoryMonitor(
        device,
        project_config.runtime.device_memory_sample_interval_seconds,
    )
    memory_monitor.start()
    overall_started = time.perf_counter()

    preparation_clock = PhaseClock(device)
    preparation_clock.start()
    source_pil = load_image(source_image)
    target_pil = load_image(target_image)
    source = pil_to_tensor(source_pil, device=device)
    target = pil_to_tensor(target_pil, device=device)
    if source.ndim != 4 or source.shape[0] != 1 or source.shape[1] != 3:
        raise ValueError("源图必须解码为单张三通道图片。")

    use_cached_centers = bool(
        selected_method is not AttackMethod.LEGACY_PIXEL
        and project_config.shared_engineering.cache_target_cluster_centers
    )
    targets = prepare_targets(
        surrogates,
        target,
        target_text,
        attack_config,
        cache=shared_target_cache,
        target_image_key=target_image_key,
        prepare_cluster_centers=use_cached_centers,
    )
    model_weights = torch.full(
        (len(surrogates),),
        1.0 / len(surrogates),
        device=device,
        dtype=source.dtype,
    )

    height, width = source.shape[-2:]
    selected_axis_ratio = float(
        project_config.frequency.initial_axis_ratio if axis_ratio is None else axis_ratio
    )
    if not 0 < selected_axis_ratio <= 1:
        raise ValueError("频率方向比例必须位于 (0, 1]。")
    progressive_schedule: ProgressiveSchedule | None = None
    progressive_basis = "not_applicable"
    if selected_method is AttackMethod.PROGRESSIVE_FREQUENCY:
        progressive = project_config.frequency.progressive
        progressive_schedule = ProgressiveSchedule(
            progressive.time_fractions,
            progressive.axis_ratios,
        )
        selected_axis_ratio = progressive.axis_ratios[0]
        progressive_basis = "elapsed_time_fraction" if time_budget_seconds else "step_fraction_fallback"

    height_frequencies = axis_frequency_count(height, selected_axis_ratio)
    width_frequencies = axis_frequency_count(width, selected_axis_ratio)
    height_basis: torch.Tensor | None = None
    width_basis: torch.Tensor | None = None
    if frequency_method:
        height_basis = shared_basis_cache.get(
            height,
            height_frequencies,
            device=device,
            dtype=source.dtype,
        )
        width_basis = shared_basis_cache.get(
            width,
            width_frequencies,
            device=device,
            dtype=source.dtype,
        )
        coefficients = source.new_zeros(1, 3, height_frequencies, width_frequencies)
    else:
        coefficients = source.new_zeros(0)
    delta = torch.zeros_like(source)
    preparation_seconds = preparation_clock.stop()

    history: list[dict[str, Any]] = []
    last_step_record: dict[str, Any] | None = None
    checkpoints: list[dict[str, Any]] = []
    captured_fractions: set[float] = set()
    radial_scales: list[float] = []
    direct_radial_scales: list[float] = []
    reprojection_iterations: list[int] = []
    initial_clipped_fractions: list[float] = []
    converged_before_fallback: list[bool] = []
    zero_gradient_steps = 0
    discarded_steps = 0
    completed_steps = 0
    step_seconds: list[float] = []
    frequency_transitions: list[dict[str, Any]] = []
    stop_reason = "requested_steps_completed"

    def current_state() -> tuple[torch.Tensor, torch.Tensor]:
        if frequency_method:
            assert height_basis is not None and width_basis is not None
            return _current_frequency_state(
                source, coefficients, height_basis, width_basis
            )
        return delta, (source + delta).clamp(0, 1)

    def capture_checkpoint(fraction: float, elapsed: float) -> None:
        _, current = current_state()
        synchronize(device)
        checkpoints.append(
            {
                "label": _checkpoint_label(fraction),
                "budget_fraction": fraction,
                "completed_steps": completed_steps,
                "loop_elapsed_seconds": elapsed,
                "axis_ratio": selected_axis_ratio if frequency_method else None,
                "height_frequencies": height_frequencies if frequency_method else None,
                "width_frequencies": width_frequencies if frequency_method else None,
                "_tensor": current.detach().cpu(),
            }
        )
        captured_fractions.add(fraction)

    synchronize(device)
    loop_started = time.perf_counter()
    if selected_method is not AttackMethod.ZERO:
        for requested_step in range(1, requested_steps + 1):
            synchronize(device)
            step_started = time.perf_counter()
            elapsed_before = step_started - loop_started
            if time_budget_seconds is not None:
                if elapsed_before >= time_budget_seconds:
                    stop_reason = "time_budget_exhausted"
                    break
                if step_seconds:
                    estimated = statistics.median(step_seconds[-5:])
                    if elapsed_before + estimated > time_budget_seconds:
                        stop_reason = "estimated_next_step_would_exceed_budget"
                        break

            if progressive_schedule is not None:
                progress = (
                    elapsed_before / time_budget_seconds
                    if time_budget_seconds is not None
                    else (requested_step - 1) / max(1, requested_steps)
                )
                scheduled_ratio = progressive_schedule.ratio_at(progress)
                new_height_frequencies = axis_frequency_count(height, scheduled_ratio)
                new_width_frequencies = axis_frequency_count(width, scheduled_ratio)
                if (
                    new_height_frequencies != height_frequencies
                    or new_width_frequencies != width_frequencies
                ):
                    old_shape = [height_frequencies, width_frequencies]
                    coefficients = expand_coefficients(
                        coefficients,
                        new_height_frequencies,
                        new_width_frequencies,
                    )
                    height_frequencies = new_height_frequencies
                    width_frequencies = new_width_frequencies
                    height_basis = shared_basis_cache.get(
                        height,
                        height_frequencies,
                        device=device,
                        dtype=source.dtype,
                    )
                    width_basis = shared_basis_cache.get(
                        width,
                        width_frequencies,
                        device=device,
                        dtype=source.dtype,
                    )
                    selected_axis_ratio = scheduled_ratio
                    frequency_transitions.append(
                        {
                            "before_step": requested_step,
                            "loop_elapsed_seconds": elapsed_before,
                            "progress_fraction": progress,
                            "old_frequency_shape": old_shape,
                            "new_frequency_shape": [height_frequencies, width_frequencies],
                            "axis_ratio": scheduled_ratio,
                        }
                    )

            weights_used = model_weights.detach().clone()
            if frequency_method:
                assert height_basis is not None and width_basis is not None
                parameter = coefficients.detach().requires_grad_(True)
                raw_graph = synthesize(parameter, height_basis, width_basis)
                adversarial_graph = (source + raw_graph).clamp(0, 1)
            else:
                parameter = delta.detach().requires_grad_(True)
                adversarial_graph = (source + parameter).clamp(0, 1)

            if selected_method is AttackMethod.LEGACY_PIXEL:
                objective = joint_objective(
                    adversarial_graph,
                    surrogates,
                    targets,
                    model_weights,
                    attack_config,
                    use_cached_target_centers=False,
                )
                parameter_gradient = torch.autograd.grad(objective.loss, parameter)[0]
            else:
                image_leaf = adversarial_graph.detach().requires_grad_(True)
                image_gradient, objective = sequential_image_gradient(
                    image_leaf,
                    surrogates,
                    targets,
                    model_weights,
                    attack_config,
                )
                parameter_gradient = torch.autograd.grad(
                    adversarial_graph,
                    parameter,
                    grad_outputs=image_gradient,
                )[0]

            if frequency_method:
                assert height_basis is not None and width_basis is not None
                update = update_coefficients(
                    coefficients,
                    parameter_gradient,
                    height_basis,
                    width_basis,
                    spatial_step_size=attack_config.spatial_step_size,
                    epsilon=attack_config.epsilon,
                    keep_constant_component=project_config.frequency.constant_component,
                    maximum_reprojection_iterations=(
                        project_config.frequency.maximum_reprojection_iterations
                    ),
                    reprojection_tolerance=project_config.frequency.reprojection_tolerance,
                )
                proposed_parameter = update.coefficients
                update_record = {
                    "zero_gradient": update.zero_gradient,
                    "direction_linf": update.direction_linf,
                    "proposed_linf": update.proposed_linf,
                    "projection_iterations": update.projection_iterations,
                    "initial_clipped_fraction": update.initial_clipped_fraction,
                    "post_reprojection_linf": update.post_reprojection_linf,
                    "converged_before_fallback": update.converged_before_fallback,
                    "direct_radial_scale": update.direct_radial_scale,
                    "radial_scale": update.radial_scale,
                    "resulting_linf": update.resulting_linf,
                }
            else:
                proposed_parameter, update_record = _pixel_update(
                    delta,
                    parameter_gradient,
                    source,
                    step_size=attack_config.spatial_step_size,
                    epsilon=attack_config.epsilon,
                )
            proposed_weights = torch.softmax(
                -objective.global_similarities.detach(), dim=0
            )
            synchronize(device)
            state_ready_elapsed = time.perf_counter() - loop_started
            if time_budget_seconds is not None and state_ready_elapsed > time_budget_seconds:
                discarded_steps += 1
                stop_reason = "time_budget_exhausted_during_step"
                break

            if frequency_method:
                coefficients = proposed_parameter
            else:
                delta = proposed_parameter
            model_weights = proposed_weights
            completed_steps += 1
            if update_record["zero_gradient"]:
                zero_gradient_steps += 1
            if update_record["radial_scale"] is not None:
                radial_scales.append(float(update_record["radial_scale"]))
                direct_radial_scales.append(float(update_record["direct_radial_scale"]))
            if update_record["projection_iterations"] is not None:
                reprojection_iterations.append(int(update_record["projection_iterations"]))
                initial_clipped_fractions.append(
                    float(update_record["initial_clipped_fraction"])
                )
                converged_before_fallback.append(
                    bool(update_record["converged_before_fallback"])
                )

            step_record = {
                "step": completed_steps,
                "requested_step": requested_step,
                **objective.detached_record(),
                "weights_used_for_this_step": _weights_list(weights_used),
                "weights_for_next_step": _weights_list(model_weights),
                "linf_after_update": update_record["resulting_linf"],
                "zero_gradient": update_record["zero_gradient"],
                "direction_linf": update_record["direction_linf"],
                "proposed_linf": update_record["proposed_linf"],
                "projection_iterations": update_record["projection_iterations"],
                "initial_clipped_fraction": update_record["initial_clipped_fraction"],
                "post_reprojection_linf": update_record["post_reprojection_linf"],
                "converged_before_fallback": update_record["converged_before_fallback"],
                "direct_radial_scale": update_record["direct_radial_scale"],
                "radial_scale": update_record["radial_scale"],
                "axis_ratio": selected_axis_ratio if frequency_method else None,
                "frequency_shape": (
                    [height_frequencies, width_frequencies] if frequency_method else None
                ),
                "state_ready_loop_seconds": state_ready_elapsed,
            }
            last_step_record = step_record
            if (
                completed_steps == 1
                or completed_steps == requested_steps
                or completed_steps % attack_config.log_every == 0
            ):
                history.append(step_record)

            for fraction in fractions:
                if fraction in captured_fractions:
                    continue
                if time_budget_seconds is None:
                    threshold_reached = completed_steps >= math.ceil(fraction * requested_steps)
                else:
                    threshold_reached = state_ready_elapsed >= fraction * time_budget_seconds
                if threshold_reached:
                    capture_checkpoint(fraction, state_ready_elapsed)

            synchronize(device)
            step_seconds.append(time.perf_counter() - step_started)
        else:
            stop_reason = "requested_steps_completed"
    else:
        stop_reason = "zero_perturbation_baseline"

    synchronize(device)
    loop_seconds = time.perf_counter() - loop_started
    if last_step_record is not None and (
        not history or history[-1]["step"] != last_step_record["step"]
    ):
        history.append(last_step_record)

    raw_perturbation, final_float = current_state()
    final_clock = PhaseClock(device)
    final_clock.start()
    final_objective = evaluate_objective(
        final_float,
        surrogates,
        targets,
        model_weights,
        attack_config,
        use_cached_target_centers=use_cached_centers,
    )
    final_recompute_seconds = final_clock.stop()

    output_clock = PhaseClock(device)
    output_clock.start()
    output_root = Path(output_directory).resolve() if output_directory is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        final_path = save_tensor_png(output_root / "adversarial.png", final_float)
        final_sha256 = sha256_file(final_path)
        decoded_pil = load_image(final_path)
    else:
        final_path = None
        final_sha256 = None
        decoded_pil = tensor_to_pil(final_float)
    decoded_tensor = pil_to_tensor(decoded_pil, device=device)

    checkpoint_records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        tensor_cpu = checkpoint.pop("_tensor")
        record = dict(checkpoint)
        checkpoint_image = tensor_to_pil(tensor_cpu)
        if output_root is not None:
            checkpoint_path = output_root / "checkpoints" / f"{record['label']}.png"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_image.save(checkpoint_path, format="PNG")
            record["image"] = str(checkpoint_path)
            record["image_sha256"] = sha256_file(checkpoint_path)
            record["decoded_audit"] = decoded_image_audit(
                source_pil,
                checkpoint_path,
                maximum_levels=project_config.acceptance.decoded_max_pixel_difference,
            )
        else:
            record["image"] = None
            record["image_sha256"] = None
            record["decoded_audit"] = decoded_image_audit(
                source_pil,
                checkpoint_image,
                maximum_levels=project_config.acceptance.decoded_max_pixel_difference,
            )
        checkpoint_records.append(record)

    decoded_audit = decoded_image_audit(
        source_pil,
        decoded_pil,
        maximum_levels=project_config.acceptance.decoded_max_pixel_difference,
    )
    if not decoded_audit["within_budget"]:
        raise RuntimeError(
            "落盘图片超过像素预算："
            f"{decoded_audit['max_absolute_pixel_difference_levels']} > "
            f"{decoded_audit['budget_levels']}。"
        )
    if height_basis is None or width_basis is None:
        height_basis = shared_basis_cache.get(
            height,
            height_frequencies,
            device=device,
            dtype=source.dtype,
        )
        width_basis = shared_basis_cache.get(
            width,
            width_frequencies,
            device=device,
            dtype=source.dtype,
        )
    diagnostics = perturbation_diagnostics(
        source,
        raw_perturbation,
        final_float,
        decoded_tensor,
        height_basis,
        width_basis,
    )
    diagnostics["diagnostic_band"] = {
        "axis_ratio": selected_axis_ratio,
        "height_frequencies": height_frequencies,
        "width_frequencies": width_frequencies,
        "meaning": (
            "optimized_frequency_band"
            if frequency_method
            else "shared_reference_band_for_spectral_diagnostics_only"
        ),
    }
    output_seconds_before_metadata = output_clock.stop()

    synchronize(device)
    total_attack_seconds_before_metadata = time.perf_counter() - overall_started
    memory = memory_monitor.stop()
    cache_after = _cache_stats_copy(shared_target_cache)
    basis_stats = shared_basis_cache.stats()
    parameter_count = (
        int(coefficients.numel()) if frequency_method else int(delta.numel())
    )
    radial_summary = {
        "count": len(radial_scales),
        "minimum": min(radial_scales) if radial_scales else None,
        "mean": statistics.fmean(radial_scales) if radial_scales else None,
        "fraction_scaled": (
            sum(value < 1.0 for value in radial_scales) / len(radial_scales)
            if radial_scales
            else None
        ),
        "counterfactual_direct_scale_mean": (
            statistics.fmean(direct_radial_scales) if direct_radial_scales else None
        ),
        "mean_fallback_minus_counterfactual_direct_scale": (
            statistics.fmean(
                actual - direct
                for actual, direct in zip(radial_scales, direct_radial_scales, strict=True)
            )
            if radial_scales
            else None
        ),
    }
    clipped_step_count = sum(value > 0 for value in reprojection_iterations)
    clipped_step_convergence = [
        converged
        for iterations, converged in zip(
            reprojection_iterations, converged_before_fallback, strict=True
        )
        if iterations > 0
    ]
    reprojection_summary = {
        "count": len(reprojection_iterations),
        "steps_with_spatial_clip": clipped_step_count,
        "fraction_of_steps_with_spatial_clip": (
            clipped_step_count / len(reprojection_iterations)
            if reprojection_iterations
            else None
        ),
        "mean_iterations": (
            statistics.fmean(reprojection_iterations) if reprojection_iterations else None
        ),
        "maximum_iterations": max(reprojection_iterations) if reprojection_iterations else None,
        "mean_initial_clipped_element_fraction": (
            statistics.fmean(initial_clipped_fractions)
            if initial_clipped_fractions
            else None
        ),
        "maximum_initial_clipped_element_fraction": (
            max(initial_clipped_fractions) if initial_clipped_fractions else None
        ),
        "fraction_clipped_steps_converged_before_fallback": (
            sum(clipped_step_convergence) / len(clipped_step_convergence)
            if clipped_step_convergence
            else None
        ),
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "success",
        "method": selected_method.value,
        "seed": project_config.seed,
        "target_text": target_text,
        "source_shape_bchw": list(source.shape),
        "target_shape_bchw": list(target.shape),
        "attack": {
            "epsilon": attack_config.epsilon,
            "spatial_step_size": attack_config.spatial_step_size,
            "steps_requested": requested_steps,
            "steps_completed": completed_steps,
            "discarded_over_budget_steps": discarded_steps,
            "stop_reason": stop_reason,
            "time_budget_seconds": time_budget_seconds,
            "clusters": attack_config.clusters,
            "cluster_iterations": attack_config.cluster_iterations,
            "sinkhorn_iterations": attack_config.sinkhorn_iterations,
            "sinkhorn_regularization": attack_config.sinkhorn_regularization,
            "sinkhorn_tolerance": attack_config.sinkhorn_tolerance,
            "detach_transport_plan": attack_config.detach_transport_plan,
            "local_weight": attack_config.local_weight,
            "text_weight": attack_config.text_weight,
            "text_model_index": attack_config.text_model_index,
            "zero_gradient_steps": zero_gradient_steps,
            "initialization": "zero",
            "gradient_execution": (
                "joint_three_surrogate_graph"
                if selected_method is AttackMethod.LEGACY_PIXEL
                else "sequential_surrogate_image_gradient"
            ),
            "target_cluster_centers_cached": use_cached_centers,
        },
        "frequency": {
            "enabled": frequency_method,
            "axis_ratio_final": selected_axis_ratio if frequency_method else None,
            "height_frequencies_final": height_frequencies if frequency_method else None,
            "width_frequencies_final": width_frequencies if frequency_method else None,
            "coefficient_count": int(coefficients.numel()) if frequency_method else None,
            "spatial_parameter_count": int(delta.numel()),
            "optimized_parameter_count": parameter_count,
            "constant_component_kept": (
                project_config.frequency.constant_component if frequency_method else None
            ),
            "projection": (
                project_config.frequency.coefficient_feasibility
                if frequency_method
                else None
            ),
            "maximum_reprojection_iterations": (
                project_config.frequency.maximum_reprojection_iterations
                if frequency_method
                else None
            ),
            "reprojection_tolerance": (
                project_config.frequency.reprojection_tolerance
                if frequency_method
                else None
            ),
            "progressive_schedule_basis": progressive_basis,
            "transitions": frequency_transitions,
            "reprojection_summary": reprojection_summary,
            "radial_scale_summary": radial_summary,
            "saved_image_claimed_strictly_bandlimited": False,
        },
        "timing": {
            "cold_model_loading_seconds": model_loading_seconds,
            "sample_preparation_seconds": preparation_seconds,
            "optimization_loop_seconds": loop_seconds,
            "final_surrogate_recompute_seconds": final_recompute_seconds,
            "final_output_seconds_before_metadata": output_seconds_before_metadata,
            "total_attack_seconds_before_metadata": total_attack_seconds_before_metadata,
            "metadata_final_write_included": False,
            "step_seconds_median": statistics.median(step_seconds) if step_seconds else None,
            "step_seconds_minimum": min(step_seconds) if step_seconds else None,
            "step_seconds_maximum": max(step_seconds) if step_seconds else None,
            "last_committed_state_ready_loop_seconds": (
                float(last_step_record["state_ready_loop_seconds"])
                if last_step_record is not None
                else 0.0
            ),
        },
        "memory": memory,
        "cache": {
            "target_before": cache_before,
            "target_after": cache_after,
            "basis": basis_stats,
        },
        "history_semantics": {
            "objective_is_before_that_step_update": True,
            "linf_is_after_that_step_update": True,
            "final_objective_is_recomputed_after_last_update": True,
        },
        "history": history,
        "final_objective": final_objective.detached_record(),
        "final_model_weights": _weights_list(model_weights),
        "perturbation_diagnostics": diagnostics,
        "decoded_image_audit": decoded_audit,
        "checkpoints": checkpoint_records,
        "output_image": str(final_path) if final_path else None,
        "output_image_sha256": final_sha256,
        "primary_evaluator_used_during_attack": False,
    }
    if output_root is not None:
        metrics_path = output_root / "attack_metrics.json"
        metadata["output_metadata"] = str(metrics_path)
        write_started = time.perf_counter()
        save_json(metrics_path, metadata)
        first_write_seconds = time.perf_counter() - write_started
        metadata["timing"]["metadata_first_write_seconds"] = first_write_seconds
        metadata["timing"]["total_attack_seconds_through_first_metadata_write"] = (
            time.perf_counter() - overall_started
        )
        # 第二次原子写只补入第一次写入的实测时间；该极小重写被明确排除，避免递归计时定义。
        save_json(metrics_path, metadata)

    return AttackResult(
        image=decoded_pil,
        tensor=final_float.detach().cpu(),
        decoded_tensor=decoded_tensor.detach().cpu(),
        metadata=metadata,
    )
