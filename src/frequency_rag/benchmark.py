from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from .config import ProjectConfig
from .io import load_json, save_json, sha256_file, stable_key
from .pipeline import utc_timestamp


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _attack_total_seconds(timing: dict[str, Any]) -> float | None:
    for key in (
        "total_attack_seconds_through_first_metadata_write",
        "total_attack_seconds_before_metadata",
    ):
        value = timing.get(key)
        if value is not None:
            return float(value)
    return None


def _artifact_json(
    run_root: Path, manifest: dict[str, Any], key: str
) -> dict[str, Any] | None:
    raw_path = manifest.get(key)
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = run_root / path
    if not path.is_file():
        return None
    expected_sha256 = manifest.get(f"{key}_sha256")
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        return None
    value = load_json(path)
    return value if isinstance(value, dict) else None


def _digest_json(value: Any) -> str:
    return stable_key([json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))])


def _run_comparability(
    run_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    config = _artifact_json(run_root, manifest, "config_snapshot")
    model = _artifact_json(run_root, manifest, "model_snapshot")
    evaluator = _artifact_json(run_root, manifest, "evaluator_snapshot")
    generation_implementation = manifest.get("implementation_content_identity")
    if not generation_implementation:
        snapshot = _artifact_json(run_root, manifest, "implementation_snapshot")
        generation_implementation = snapshot.get("content_identity") if snapshot else None
    evaluation_implementation = manifest.get("evaluation_implementation_content_identity")
    if not evaluation_implementation:
        snapshot = _artifact_json(run_root, manifest, "evaluation_implementation_snapshot")
        evaluation_implementation = snapshot.get("content_identity") if snapshot else None

    stable_surrogates: list[dict[str, Any]] | None = None
    stable_hardware: dict[str, Any] | None = None
    if model is not None and isinstance(model.get("surrogates"), list):
        surrogate_fields = (
            "name",
            "pretrained",
            "hf_repo",
            "weight_sha256",
            "weight_revision",
            "device",
            "precision",
            "image_size",
            "preprocess",
            "open_clip_torch_version",
            "requires_true_local_tokens",
        )
        stable_surrogates = [
            {field: item.get(field) for field in surrogate_fields}
            for item in model["surrogates"]
            if isinstance(item, dict)
        ]
        stable_hardware = {
            "torch_version": model.get("torch_version"),
            "cuda_version": model.get("cuda_version"),
            "device_name": model.get("device_name"),
        }

    stable_evaluator: dict[str, Any] | None = None
    if evaluator is not None:
        evaluator_fields = (
            "encoder_name",
            "model_id",
            "revision",
            "device",
            "precision",
            "transformers_version",
            "processor_class",
            "processor_use_fast",
            "model_class",
        )
        stable_evaluator = {field: evaluator.get(field) for field in evaluator_fields}
        stable_evaluator["weights"] = [
            {"sha256": item.get("sha256"), "bytes": item.get("bytes")}
            for item in evaluator.get("weight_files", [])
            if isinstance(item, dict)
        ]

    components = {
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "configuration": _digest_json(config) if config is not None else None,
        "surrogates": _digest_json(stable_surrogates) if stable_surrogates is not None else None,
        "hardware_and_framework": (
            _digest_json(stable_hardware) if stable_hardware is not None else None
        ),
        "evaluator": _digest_json(stable_evaluator) if stable_evaluator is not None else None,
        "generation_implementation": generation_implementation,
        "evaluation_implementation": evaluation_implementation,
    }
    missing = [key for key, value in components.items() if not value]
    return {
        "complete": not missing,
        "missing_components": missing,
        "components": components,
        "signature": _digest_json(components) if not missing else None,
    }


def generation_conditions_signature(
    run_root: str | Path, manifest: dict[str, Any] | None = None
) -> str:
    """返回生成阶段公平计时条件签名；缺少证据时拒绝继续。"""
    root = Path(run_root).resolve()
    run_manifest = manifest or load_json(root / "run_manifest.json")
    comparability = _run_comparability(root, run_manifest)
    required_components = (
        "source_manifest_sha256",
        "configuration",
        "surrogates",
        "hardware_and_framework",
        "generation_implementation",
    )
    selected = {
        key: comparability["components"].get(key) for key in required_components
    }
    missing = [key for key, value in selected.items() if not value]
    if missing:
        raise ValueError(
            "公平像素参考运行缺少生成条件证据：" + "、".join(missing)
        )
    return _digest_json(selected)


def collect_result_rows(run_directories: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_directory in run_directories:
        run_root = Path(run_directory).resolve()
        manifest = load_json(run_root / "run_manifest.json")
        method = str(manifest.get("method", "unknown"))
        comparability = _run_comparability(run_root, manifest)
        for sample in manifest.get("samples", []):
            row: dict[str, Any] = {
                "run_directory": str(run_root),
                "method": method,
                "sample_id": sample.get("sample_id"),
                "status": sample.get("status"),
                "evaluation_status": sample.get("evaluation_status"),
                "source_image": sample.get("source_image"),
                "source_sha256": sample.get("source_sha256"),
                "target_image": sample.get("target_image"),
                "target_sha256": sample.get("target_sha256"),
                "question": sample.get("question"),
                "target_text": sample.get("target_text"),
                "failure": sample.get("failure") or sample.get("evaluation_failure"),
                "run_comparability_complete": comparability["complete"],
                "run_comparability_missing_components": comparability[
                    "missing_components"
                ],
                "run_comparability_components": comparability["components"],
                "run_comparability_signature": comparability["signature"],
            }
            artifact_integrity: dict[str, dict[str, Any]] = {}
            for artifact_name, path_key, hash_key in (
                ("attack_metrics", "attack_metrics", "attack_metrics_sha256"),
                ("vector_evaluation", "vector_evaluation", "vector_evaluation_sha256"),
            ):
                artifact_path = Path(sample[path_key]) if sample.get(path_key) else None
                if artifact_path is not None and not artifact_path.is_absolute():
                    artifact_path = run_root / artifact_path
                expected_hash = sample.get(hash_key)
                actual_hash = (
                    sha256_file(artifact_path)
                    if artifact_path is not None and artifact_path.is_file()
                    else None
                )
                artifact_integrity[artifact_name] = {
                    "path": str(artifact_path) if artifact_path is not None else None,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "matches": bool(expected_hash and actual_hash == expected_hash),
                }
            row["artifact_integrity"] = artifact_integrity
            row["artifact_integrity_complete"] = all(
                item["matches"] for item in artifact_integrity.values()
            )
            attack_integrity = artifact_integrity["attack_metrics"]
            if attack_integrity["matches"]:
                attack = load_json(attack_integrity["path"])
                timing = attack.get("timing", {})
                row.update(
                    {
                        "source_shape_bchw": attack.get("source_shape_bchw"),
                        "steps_requested": _nested(attack, "attack", "steps_requested"),
                        "steps_completed": _nested(attack, "attack", "steps_completed"),
                        "time_budget_seconds": _nested(attack, "attack", "time_budget_seconds"),
                        "stop_reason": _nested(attack, "attack", "stop_reason"),
                        "axis_ratio": _nested(attack, "frequency", "axis_ratio_final"),
                        "frequency_shape": [
                            _nested(attack, "frequency", "height_frequencies_final"),
                            _nested(attack, "frequency", "width_frequencies_final"),
                        ] if _nested(attack, "frequency", "enabled", default=False) else None,
                        "optimized_parameter_count": _nested(attack, "frequency", "optimized_parameter_count"),
                        "loop_seconds": timing.get("optimization_loop_seconds"),
                        "last_committed_state_ready_loop_seconds": timing.get(
                            "last_committed_state_ready_loop_seconds"
                        ),
                        "preparation_seconds": timing.get("sample_preparation_seconds"),
                        "final_surrogate_recompute_seconds": timing.get(
                            "final_surrogate_recompute_seconds"
                        ),
                        "output_seconds": timing.get("final_output_seconds_before_metadata"),
                        "total_attack_seconds": _attack_total_seconds(timing),
                        "peak_allocated_bytes": _nested(
                            attack, "memory", "framework_peak_allocated_bytes"
                        ),
                        "peak_reserved_bytes": _nested(
                            attack, "memory", "framework_peak_reserved_bytes"
                        ),
                        "device_peak_sampled_bytes": _nested(
                            attack, "memory", "device_used_peak_sampled_bytes"
                        ),
                        "decoded_max_pixel_difference_levels": _nested(
                            attack,
                            "decoded_image_audit",
                            "max_absolute_pixel_difference_levels",
                        ),
                        "decoded_mean_absolute_pixel_difference_levels": _nested(
                            attack,
                            "decoded_image_audit",
                            "mean_absolute_pixel_difference_levels",
                        ),
                        "decoded_mean_squared_pixel_difference_levels": _nested(
                            attack,
                            "decoded_image_audit",
                            "mean_squared_pixel_difference_levels",
                        ),
                        "range_clipped_element_fraction": _nested(
                            attack,
                            "perturbation_diagnostics",
                            "range_clipped_element_fraction",
                        ),
                        "out_of_band_energy_ratio_after_png": _nested(
                            attack,
                            "perturbation_diagnostics",
                            "after_png_reload",
                            "out_of_band_energy_ratio",
                        ),
                        "frequency_projection": _nested(
                            attack,
                            "frequency",
                            "projection",
                        ),
                        "reprojection_iterations_mean": _nested(
                            attack,
                            "frequency",
                            "reprojection_summary",
                            "mean_iterations",
                        ),
                        "fraction_of_steps_with_spatial_clip": _nested(
                            attack,
                            "frequency",
                            "reprojection_summary",
                            "fraction_of_steps_with_spatial_clip",
                        ),
                        "fallback_radial_scale_mean": _nested(
                            attack,
                            "frequency",
                            "radial_scale_summary",
                            "mean",
                        ),
                        "radial_scale_mean": _nested(
                            attack,
                            "frequency",
                            "radial_scale_summary",
                            "mean",
                        ),
                        "checkpoints": attack.get("checkpoints", []),
                    }
                )
            evaluation_integrity = artifact_integrity["vector_evaluation"]
            if evaluation_integrity["matches"]:
                evaluation = load_json(evaluation_integrity["path"])
                cosine = evaluation.get("cosine_metrics", {})
                distance = evaluation.get("distance_metrics", {})
                row.update(cosine)
                row.update(distance)
                row["embedding_dimension"] = evaluation.get("embedding_dimension")
                row["checkpoint_evaluations"] = evaluation.get("checkpoints", [])
                row["evaluation_seconds"] = _nested(evaluation, "cost", "evaluation_seconds")
                row["evaluation_peak_allocated_bytes"] = _nested(
                    evaluation,
                    "cost",
                    "peak_memory",
                    "framework_peak_allocated_bytes",
                )
            rows.append(row)
    return rows


def _finite_mean(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None and np.isfinite(value)]
    return statistics.fmean(selected) if selected else None


def _finite_median(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None and np.isfinite(value)]
    return statistics.median(selected) if selected else None


def _finite_range(values: Iterable[float | int | None]) -> dict[str, float | None]:
    selected = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "minimum": min(selected) if selected else None,
        "maximum": max(selected) if selected else None,
    }


def _aggregate_method(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if row.get("evaluation_status") == "success"
        and row.get("artifact_integrity_complete")
    ]
    generation_success = [
        row
        for row in rows
        if row.get("status") in {"generated", "evaluated"}
        and _nested(row, "artifact_integrity", "attack_metrics", "matches", default=False)
    ]
    numeric_fields = (
        "target_cosine_gain",
        "query_cosine_gain",
        "source_adversarial_cosine",
        "loop_seconds",
        "total_attack_seconds",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "decoded_max_pixel_difference_levels",
        "decoded_mean_absolute_pixel_difference_levels",
        "decoded_mean_squared_pixel_difference_levels",
        "out_of_band_energy_ratio_after_png",
    )
    return {
        "method": method,
        "planned_rows": len(rows),
        "generation_success_rows": len(generation_success),
        "evaluated_success_rows": len(successful),
        "failed_rows": len(rows) - len(successful),
        "means": {
            field: _finite_mean(row.get(field) for row in successful)
            for field in numeric_fields
        },
        "medians": {
            field: _finite_median(row.get(field) for row in successful)
            for field in numeric_fields
        },
        "timing_ranges": {
            field: _finite_range(row.get(field) for row in successful)
            for field in ("loop_seconds", "total_attack_seconds")
        },
    }


def _representative_by_sample(rows: list[dict[str, Any]], method: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (
            row.get("method") == method
            and row.get("evaluation_status") == "success"
            and row.get("artifact_integrity_complete")
        ):
            grouped.setdefault(str(row["sample_id"]), []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for sample_id, repetitions in grouped.items():
        ordered = sorted(
            repetitions,
            key=lambda row: (
                float("inf") if row.get("total_attack_seconds") is None else row["total_attack_seconds"]
            ),
        )
        representative = dict(ordered[len(ordered) // 2])
        distinct_run_directories = {
            str(row.get("run_directory")) for row in repetitions
        }
        representative["timing_repetitions_found"] = len(distinct_run_directories)
        representative["successful_rows_found"] = len(repetitions)
        representative["duplicate_success_rows_within_runs"] = (
            len(repetitions) - len(distinct_run_directories)
        )
        sample_identities = {
            (
                row.get("source_sha256"),
                row.get("target_sha256"),
                row.get("question"),
                row.get("target_text"),
            )
            for row in repetitions
        }
        comparability_signatures = {
            str(row["run_comparability_signature"])
            for row in repetitions
            if row.get("run_comparability_signature")
        }
        all_comparability_complete = all(
            bool(row.get("run_comparability_complete")) for row in repetitions
        )
        representative["repetition_sample_identity_consistent"] = (
            len(sample_identities) == 1
        )
        experiment_settings = {
            (
                row.get("steps_requested"),
                row.get("time_budget_seconds"),
                row.get("axis_ratio"),
                tuple(row.get("frequency_shape") or ()),
            )
            for row in repetitions
        }
        representative["repetition_experiment_settings_consistent"] = (
            len(experiment_settings) == 1
        )
        representative["repetition_run_conditions_comparable"] = bool(
            all_comparability_complete and len(comparability_signatures) == 1
        )
        representative["repetition_run_comparability_signatures"] = sorted(
            comparability_signatures
        )
        median_fields = (
            "source_target_cosine",
            "adversarial_target_cosine",
            "target_cosine_gain",
            "adversarial_query_cosine",
            "query_cosine_gain",
            "source_adversarial_cosine",
            "preparation_seconds",
            "loop_seconds",
            "last_committed_state_ready_loop_seconds",
            "final_surrogate_recompute_seconds",
            "output_seconds",
            "total_attack_seconds",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
        for field in median_fields:
            median = _finite_median(row.get(field) for row in repetitions)
            representative[f"median_{field}_across_repetitions"] = median
            if median is not None:
                representative[field] = median
        result[sample_id] = representative
    return result


def _bootstrap_source_group_mean_interval(
    pairs: list[dict[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any] | None:
    grouped: dict[str, list[float]] = {}
    for pair in pairs:
        value = pair.get("target_gain_advantage")
        source = pair.get("source_sha256")
        if value is not None and source:
            grouped.setdefault(str(source), []).append(float(value))
    if not grouped:
        return None
    group_means = np.asarray(
        [statistics.fmean(values) for values in grouped.values()], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    samples = generator.choice(
        group_means,
        size=(iterations, len(group_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "unique_source_groups": int(len(group_means)),
        "iterations": iterations,
        "seed": seed,
        "paired_mean_advantage": float(group_means.mean()),
        "percentile_95_interval": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "interpretation": "历史兼容集合规模小且目标重复，此区间仅作探索描述。",
    }


def _paired_assessment(
    rows: list[dict[str, Any]],
    project_config: ProjectConfig,
) -> dict[str, Any]:
    pixel = _representative_by_sample(rows, "fair_pixel")
    frequency = _representative_by_sample(rows, "frequency")
    pixel_planned = {
        str(row["sample_id"])
        for row in rows
        if row.get("method") == "fair_pixel" and row.get("sample_id") is not None
    }
    frequency_planned = {
        str(row["sample_id"])
        for row in rows
        if row.get("method") == "frequency" and row.get("sample_id") is not None
    }
    # 使用两组计划样本的并集作为分母；某组少跑一行不能靠交集被静默删除。
    planned_shared_ids = sorted(pixel_planned | frequency_planned)
    candidate_shared_ids = sorted(set(pixel) & set(frequency))
    identity_mismatch_ids = sorted(
        sample_id
        for sample_id in candidate_shared_ids
        if (
            not pixel[sample_id].get("repetition_sample_identity_consistent")
            or not frequency[sample_id].get("repetition_sample_identity_consistent")
            or (
                pixel[sample_id].get("source_sha256"),
                pixel[sample_id].get("target_sha256"),
                pixel[sample_id].get("question"),
                pixel[sample_id].get("target_text"),
            )
            != (
                frequency[sample_id].get("source_sha256"),
                frequency[sample_id].get("target_sha256"),
                frequency[sample_id].get("question"),
                frequency[sample_id].get("target_text"),
            )
        )
    )
    shared_ids = sorted(set(candidate_shared_ids) - set(identity_mismatch_ids))
    failed_or_missing_ids = sorted(set(planned_shared_ids) - set(shared_ids))
    pairs: list[dict[str, Any]] = []
    equivalent_ratios: list[float] = []
    reached_count = 0
    initial_reached_count = 0
    for sample_id in shared_ids:
        pixel_row = pixel[sample_id]
        frequency_row = frequency[sample_id]
        target_advantage = (
            float(frequency_row["target_cosine_gain"])
            - float(pixel_row["target_cosine_gain"])
        )
        pixel_total_seconds = pixel_row.get("total_attack_seconds")
        frequency_total_seconds = frequency_row.get("total_attack_seconds")
        total_ratio = None
        if pixel_total_seconds and frequency_total_seconds is not None:
            total_ratio = float(frequency_total_seconds) / float(pixel_total_seconds)
        pixel_peak = pixel_row.get("peak_allocated_bytes")
        frequency_peak = frequency_row.get("peak_allocated_bytes")
        memory_ratio = None
        if pixel_peak and frequency_peak is not None:
            memory_ratio = float(frequency_peak) / float(pixel_peak)

        threshold = float(pixel_row["adversarial_target_cosine"]) - (
            project_config.acceptance.equivalent_effect_cosine_tolerance
        )
        source_similarity = float(frequency_row["source_target_cosine"])
        reached = False
        initial_reached = source_similarity >= threshold
        reach_time = 0.0 if initial_reached else None
        reaching_checkpoint = None
        if initial_reached:
            reached = True
            initial_reached_count += 1
        else:
            checkpoints = sorted(
                frequency_row.get("checkpoint_evaluations", []),
                key=lambda value: float(value.get("loop_elapsed_seconds", float("inf"))),
            )
            for checkpoint in checkpoints:
                similarity = _nested(
                    checkpoint,
                    "vector_metrics",
                    "cosine_metrics",
                    "adversarial_target_cosine",
                )
                if similarity is not None and float(similarity) >= threshold:
                    reached = True
                    reach_time = float(checkpoint["loop_elapsed_seconds"])
                    reaching_checkpoint = checkpoint.get("label")
                    break
            if (
                not reached
                and frequency_row.get("adversarial_target_cosine") is not None
                and float(frequency_row["adversarial_target_cosine"]) >= threshold
            ):
                reached = True
                reach_time = float(
                    frequency_row.get("last_committed_state_ready_loop_seconds")
                    if frequency_row.get("last_committed_state_ready_loop_seconds") is not None
                    else frequency_row.get("loop_seconds") or 0
                )
                reaching_checkpoint = "final_output"
        equivalent_total_ratio = None
        if reached:
            reached_count += 1
            if not initial_reached and pixel_total_seconds:
                estimated_total = (
                    float(frequency_row.get("preparation_seconds") or 0)
                    + float(reach_time or 0)
                    + float(frequency_row.get("final_surrogate_recompute_seconds") or 0)
                    + float(frequency_row.get("output_seconds") or 0)
                )
                equivalent_total_ratio = estimated_total / float(pixel_total_seconds)
                equivalent_ratios.append(equivalent_total_ratio)

        budget = frequency_row.get("time_budget_seconds")
        pixel_loop = pixel_row.get("loop_seconds")
        required_repetitions = max(1, int(project_config.experiment.timing_repetitions))
        pixel_repetitions = int(pixel_row.get("timing_repetitions_found") or 0)
        frequency_repetitions = int(frequency_row.get("timing_repetitions_found") or 0)
        repetitions_sufficient = bool(
            pixel_repetitions >= required_repetitions
            and frequency_repetitions >= required_repetitions
        )
        run_conditions_comparable = bool(
            pixel_row.get("repetition_run_conditions_comparable")
            and frequency_row.get("repetition_run_conditions_comparable")
            and pixel_row.get("repetition_experiment_settings_consistent")
            and frequency_row.get("repetition_experiment_settings_consistent")
            and not pixel_row.get("duplicate_success_rows_within_runs")
            and not frequency_row.get("duplicate_success_rows_within_runs")
            and pixel_row.get("repetition_run_comparability_signatures")
            == frequency_row.get("repetition_run_comparability_signatures")
        )
        equal_time_compatible = bool(
            budget is not None
            and pixel_loop
            and abs(float(budget) - float(pixel_loop)) / float(pixel_loop) <= 0.05
        )
        pairs.append(
            {
                "sample_id": sample_id,
                "source_sha256": pixel_row.get("source_sha256"),
                "target_gain_advantage": target_advantage,
                "query_cosine_difference": (
                    float(frequency_row["adversarial_query_cosine"])
                    - float(pixel_row["adversarial_query_cosine"])
                ),
                "source_adversarial_cosine_difference": (
                    float(frequency_row["source_adversarial_cosine"])
                    - float(pixel_row["source_adversarial_cosine"])
                ),
                "total_attack_time_ratio": total_ratio,
                "peak_allocated_memory_ratio": memory_ratio,
                "equal_time_budget_compatible": equal_time_compatible,
                "required_timing_repetitions": required_repetitions,
                "fair_pixel_successful_repetitions": pixel_repetitions,
                "frequency_successful_repetitions": frequency_repetitions,
                "timing_repetitions_sufficient": repetitions_sufficient,
                "run_conditions_comparable": run_conditions_comparable,
                "equivalent_effect_threshold": threshold,
                "equivalent_effect_reached": reached,
                "initial_source_already_reached": initial_reached,
                "first_reaching_checkpoint": reaching_checkpoint,
                "first_reaching_loop_seconds": reach_time,
                "estimated_total_attack_time_ratio_at_reach": equivalent_total_ratio,
            }
        )

    all_repetitions_sufficient = bool(planned_shared_ids) and not failed_or_missing_ids and all(
        pair["timing_repetitions_sufficient"] for pair in pairs
    )
    all_run_conditions_comparable = (
        bool(planned_shared_ids)
        and not failed_or_missing_ids
        and all(pair["run_conditions_comparable"] for pair in pairs)
    )
    all_equal_time = all_repetitions_sufficient and all_run_conditions_comparable and all(
        pair["equal_time_budget_compatible"] for pair in pairs
    )
    mean_advantage = _finite_mean(pair["target_gain_advantage"] for pair in pairs)
    mean_query_difference = _finite_mean(pair["query_cosine_difference"] for pair in pairs)
    mean_source_adv_difference = _finite_mean(
        pair["source_adversarial_cosine_difference"] for pair in pairs
    )
    max_pixel_ok = not failed_or_missing_ids and all(
        row.get("decoded_max_pixel_difference_levels") is not None
        and int(row["decoded_max_pixel_difference_levels"])
        <= project_config.acceptance.decoded_max_pixel_difference
        for row in [frequency[sample_id] for sample_id in shared_ids]
    ) if shared_ids else False
    quality_ok = bool(
        mean_query_difference is not None
        and mean_query_difference >= -project_config.acceptance.mean_query_cosine_drop_max
        and mean_source_adv_difference is not None
        and mean_source_adv_difference
        >= -project_config.acceptance.mean_source_adversarial_cosine_drop_max
        and max_pixel_ok
    )
    median_total_ratio = _finite_median(pair["total_attack_time_ratio"] for pair in pairs)
    memory_ratios = [
        float(pair["peak_allocated_memory_ratio"])
        for pair in pairs
        if pair.get("peak_allocated_memory_ratio") is not None
    ]
    median_memory_ratio = statistics.median(memory_ratios) if memory_ratios else None
    maximum_memory_ratio = max(memory_ratios) if memory_ratios else None
    same_time_pass = bool(
        all_equal_time
        and mean_advantage is not None
        and mean_advantage >= project_config.acceptance.same_time_mean_target_gain_advantage
        and median_total_ratio is not None
        and median_total_ratio <= project_config.acceptance.same_time_total_attack_ratio_max
        and maximum_memory_ratio is not None
        and maximum_memory_ratio <= project_config.acceptance.same_time_peak_memory_ratio_max
        and quality_ok
    )
    reach_rate = reached_count / len(planned_shared_ids) if planned_shared_ids else None
    equivalent_median = statistics.median(equivalent_ratios) if equivalent_ratios else None
    equivalent_pass = bool(
        all_repetitions_sufficient
        and all_run_conditions_comparable
        and reach_rate is not None
        and reach_rate >= project_config.acceptance.equivalent_effect_reach_rate_min
        and equivalent_median is not None
        and equivalent_median
        <= project_config.acceptance.equivalent_effect_median_total_attack_time_ratio_max
        and quality_ok
    )
    return {
        "status": "assessed" if planned_shared_ids else "not_assessable_missing_paired_runs",
        "planned_paired_sample_count": len(planned_shared_ids),
        "paired_sample_count": len(pairs),
        "failed_or_missing_paired_sample_ids": failed_or_missing_ids,
        "sample_identity_mismatch_ids": identity_mismatch_ids,
        "timing_repetitions_required": max(
            1, int(project_config.experiment.timing_repetitions)
        ),
        "all_paired_timing_repetitions_sufficient": all_repetitions_sufficient,
        "all_paired_run_conditions_comparable": all_run_conditions_comparable,
        "pairs": pairs,
        "quality_protection": {
            "mean_query_cosine_difference_frequency_minus_pixel": mean_query_difference,
            "minimum_allowed": -project_config.acceptance.mean_query_cosine_drop_max,
            "mean_source_adversarial_cosine_difference_frequency_minus_pixel": mean_source_adv_difference,
            "source_similarity_minimum_allowed": -project_config.acceptance.mean_source_adversarial_cosine_drop_max,
            "all_frequency_pngs_within_pixel_budget": max_pixel_ok,
            "passed": quality_ok,
        },
        "same_time_route": {
            "budgets_verified_compatible": all_equal_time,
            "timing_repetitions_sufficient": all_repetitions_sufficient,
            "run_conditions_comparable": all_run_conditions_comparable,
            "paired_mean_target_gain_advantage": mean_advantage,
            "required_advantage": project_config.acceptance.same_time_mean_target_gain_advantage,
            "median_total_attack_time_ratio": median_total_ratio,
            "maximum_time_ratio": project_config.acceptance.same_time_total_attack_ratio_max,
            "median_peak_allocated_memory_ratio": median_memory_ratio,
            "maximum_peak_allocated_memory_ratio": maximum_memory_ratio,
            "maximum_memory_ratio": project_config.acceptance.same_time_peak_memory_ratio_max,
            "passed": same_time_pass,
        },
        "equivalent_effect_route": {
            "timing_repetitions_sufficient": all_repetitions_sufficient,
            "run_conditions_comparable": all_run_conditions_comparable,
            "cosine_tolerance": project_config.acceptance.equivalent_effect_cosine_tolerance,
            "reach_rate": reach_rate,
            "required_reach_rate": project_config.acceptance.equivalent_effect_reach_rate_min,
            "initial_source_already_reached_count": initial_reached_count,
            "median_estimated_total_attack_time_ratio_reached_noninitial": equivalent_median,
            "maximum_ratio": project_config.acceptance.equivalent_effect_median_total_attack_time_ratio_max,
            "passed": equivalent_pass,
            "note": "检查点达标时间来自离线评估，不代表攻击程序在线知道停止时刻。",
        },
        "source_group_bootstrap": _bootstrap_source_group_mean_interval(pairs),
        "either_predeclared_route_passed": same_time_pass or equivalent_pass,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _collect_run_costs(run_paths: Iterable[Path]) -> list[dict[str, Any]]:
    costs: list[dict[str, Any]] = []
    for run_root in run_paths:
        manifest = load_json(run_root / "run_manifest.json")
        costs.append(
            {
                "run_directory": str(run_root),
                "method": manifest.get("method"),
                "stage": manifest.get("stage"),
                "generation_status": manifest.get("status"),
                "evaluation_status": manifest.get("evaluation_status"),
                "planned_sample_count": manifest.get("planned_sample_count"),
                "generated_sample_count": manifest.get("generated_sample_count"),
                "evaluated_sample_count": manifest.get("evaluated_sample_count"),
                "generation_failed_sample_count": manifest.get("failed_sample_count"),
                "evaluation_failed_sample_count": manifest.get(
                    "evaluation_failed_sample_count"
                ),
                "evaluation_skipped_generation_failure_count": manifest.get(
                    "evaluation_skipped_generation_failure_count"
                ),
                "manifest_and_image_hash_validation_seconds": _nested(
                    manifest,
                    "selection_stage",
                    "manifest_and_image_hash_validation_seconds",
                ),
                "cold_surrogate_loading_seconds": manifest.get(
                    "cold_model_loading_seconds"
                ),
                "cold_surrogate_loading_peak_allocated_bytes": _nested(
                    manifest,
                    "cold_model_loading_memory",
                    "framework_peak_allocated_bytes",
                ),
                "generation_wall_seconds": manifest.get(
                    "generation_wall_seconds_including_manifest_validation_and_model_load"
                ),
                "cold_evaluator_loading_seconds": manifest.get(
                    "cold_evaluator_loading_seconds"
                ),
                "cold_evaluator_loading_peak_allocated_bytes": _nested(
                    manifest,
                    "cold_evaluator_loading_memory",
                    "framework_peak_allocated_bytes",
                ),
                "evaluation_wall_seconds": manifest.get(
                    "evaluation_wall_seconds_including_model_load"
                ),
                "measured_pipeline_seconds_excluding_command_idle": manifest.get(
                    "measured_pipeline_stage_seconds_excluding_idle_between_commands"
                ),
                "generation_implementation_content_identity": manifest.get(
                    "implementation_content_identity"
                ),
                "evaluation_implementation_content_identity": manifest.get(
                    "evaluation_implementation_content_identity"
                ),
                "time_budget_provenance": manifest.get("time_budget_provenance"),
            }
        )
    return costs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_summary(
    run_directories: Iterable[str | Path],
    *,
    output_directory: str | Path,
    project_config: ProjectConfig,
) -> dict[str, Any]:
    project_config.validate()
    run_paths = [Path(path).resolve() for path in run_directories]
    if not run_paths:
        raise ValueError("至少需要一个运行目录才能汇总。")
    if len(set(run_paths)) != len(run_paths):
        raise ValueError("汇总运行目录包含重复路径，拒绝把同一次运行冒充重复计时。")
    rows = collect_result_rows(run_paths)
    if not rows:
        raise ValueError("运行清单中没有可汇总的样本行。")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows})
    aggregates = [
        _aggregate_method(method, [row for row in rows if row["method"] == method])
        for method in methods
    ]
    run_costs = _collect_run_costs(run_paths)
    run_cost_aggregates: list[dict[str, Any]] = []
    for method in methods:
        selected_costs = [row for row in run_costs if row.get("method") == method]
        run_cost_aggregates.append(
            {
            "method": method,
            "run_count": len(selected_costs),
            "medians": {
                field: _finite_median(row.get(field) for row in selected_costs)
                for field in (
                    "cold_surrogate_loading_seconds",
                    "generation_wall_seconds",
                    "cold_evaluator_loading_seconds",
                    "evaluation_wall_seconds",
                    "measured_pipeline_seconds_excluding_command_idle",
                )
            },
            "ranges": {
                field: _finite_range(row.get(field) for row in selected_costs)
                for field in (
                    "cold_surrogate_loading_seconds",
                    "generation_wall_seconds",
                    "cold_evaluator_loading_seconds",
                    "evaluation_wall_seconds",
                    "measured_pipeline_seconds_excluding_command_idle",
                )
            },
            }
        )
    assessment = _paired_assessment(rows, project_config)
    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "run_directories": [str(path) for path in run_paths],
        "row_count": len(rows),
        "methods": methods,
        "aggregates": aggregates,
        "run_costs": run_costs,
        "run_cost_aggregates": run_cost_aggregates,
        "predeclared_acceptance_assessment": assessment,
        "conclusion": (
            "达到至少一条预设继续研究路线。"
            if assessment["either_predeclared_route_passed"]
            else "当前结果未证明达到预设继续研究门槛，或条件不足以评估。"
        ),
        "rows": rows,
    }
    save_json(output / "summary.json", report)
    result_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"checkpoints", "checkpoint_evaluations"}
        }
        for row in rows
    ]
    _write_csv(output / "results.csv", result_rows)
    _write_csv(output / "run_costs.csv", run_costs)
    return report
