from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frequency_rag.evaluation.results import write_summary
from frequency_rag.cli import _time_budgets_from_runs
from frequency_rag.evaluation.vectors import compute_vector_metrics
from frequency_rag.common.io import load_json, save_json, sha256_file


def test_vector_metrics_preserve_original_fields_and_unit_vector_identity() -> None:
    source = np.asarray([1.0, 0.0], dtype=np.float32)
    adversarial = np.asarray([0.8, 0.6], dtype=np.float32)
    target = np.asarray([0.0, 1.0], dtype=np.float32)
    query = np.asarray([1.0, 1.0], dtype=np.float32)
    result = compute_vector_metrics(
        sample_id="sample",
        encoder_name="encoder",
        encoder_model_id="model",
        query_text="query",
        source_vector=source,
        adversarial_vector=adversarial,
        target_vector=target,
        query_vector=query,
    )
    assert set(result["cosine_metrics"]) == {
        "source_target_cosine",
        "adversarial_target_cosine",
        "target_cosine_gain",
        "source_query_cosine",
        "adversarial_query_cosine",
        "query_cosine_gain",
        "source_adversarial_cosine",
    }
    cosine = result["cosine_metrics"]["adversarial_target_cosine"]
    distance = result["distance_metrics"]["adversarial_target_l2_distance"]
    assert distance == pytest.approx(np.sqrt(2 - 2 * cosine), abs=1e-7)


def _write_run(
    root: Path,
    *,
    method: str,
    target_gain: float,
    adversarial_target: float,
    total_seconds: float,
    loop_seconds: float,
    memory_bytes: int,
    time_budget: float | None,
    checkpoint_target: float,
) -> None:
    sample_dir = root / "sample_01"
    sample_dir.mkdir(parents=True)
    attack_path = sample_dir / "attack_metrics.json"
    evaluation_path = sample_dir / "vector_evaluation.json"
    attack = {
        "source_shape_bchw": [1, 3, 5, 7],
        "attack": {
            "steps_requested": 1000,
            "steps_completed": 800 if time_budget is not None else 1000,
            "time_budget_seconds": time_budget,
            "stop_reason": "time_budget_exhausted" if time_budget else "requested_steps_completed",
        },
        "frequency": {
            "enabled": method == "frequency",
            "projection": (
                "spatial_clip_low_frequency_reproject_with_radial_fallback"
                if method == "frequency"
                else None
            ),
            "axis_ratio_final": 0.25 if method == "frequency" else None,
            "height_frequencies_final": 2 if method == "frequency" else None,
            "width_frequencies_final": 2 if method == "frequency" else None,
            "optimized_parameter_count": 12 if method == "frequency" else 105,
            "reprojection_summary": {
                "mean_iterations": 1.0 if method == "frequency" else None,
                "fraction_of_steps_with_spatial_clip": (
                    0.25 if method == "frequency" else None
                ),
            },
            "radial_scale_summary": {"mean": 1.0 if method == "frequency" else None},
        },
        "timing": {
            "sample_preparation_seconds": 1.0,
            "optimization_loop_seconds": loop_seconds,
            "final_output_seconds_before_metadata": 1.0,
            "total_attack_seconds_through_first_metadata_write": total_seconds,
        },
        "memory": {
            "framework_peak_allocated_bytes": memory_bytes,
            "framework_peak_reserved_bytes": memory_bytes + 100,
            "device_used_peak_sampled_bytes": memory_bytes + 200,
        },
        "decoded_image_audit": {
            "max_absolute_pixel_difference_levels": 16,
            "mean_absolute_pixel_difference_levels": 2.0,
            "mean_squared_pixel_difference_levels": 5.0,
        },
        "perturbation_diagnostics": {
            "range_clipped_element_fraction": 0.0,
            "after_png_reload": {"out_of_band_energy_ratio": 0.01},
        },
        "checkpoints": [{"label": "budget_0500"}],
    }
    evaluation = {
        "embedding_dimension": 512,
        "cosine_metrics": {
            "source_target_cosine": 0.4,
            "adversarial_target_cosine": adversarial_target,
            "target_cosine_gain": target_gain,
            "source_query_cosine": 0.3,
            "adversarial_query_cosine": 0.31,
            "query_cosine_gain": 0.01,
            "source_adversarial_cosine": 0.95,
        },
        "distance_metrics": {
            "source_target_l2_distance": 1.0,
            "adversarial_target_l2_distance": 0.9,
            "target_l2_distance_reduction": 0.1,
        },
        "checkpoints": [
            {
                "label": "budget_0500",
                "loop_elapsed_seconds": 4.0,
                "vector_metrics": {
                    "cosine_metrics": {
                        "adversarial_target_cosine": checkpoint_target,
                    }
                },
            }
        ],
        "cost": {
            "evaluation_seconds": 1.0,
            "peak_memory": {"framework_peak_allocated_bytes": 500},
        },
    }
    save_json(attack_path, attack)
    save_json(evaluation_path, evaluation)
    config_path = root / "config_snapshot.json"
    model_path = root / "model_snapshot.json"
    evaluator_path = root / "evaluator_snapshot.json"
    save_json(config_path, {"seed": 42, "device": "cpu", "attack_precision": "fp32"})
    save_json(
        model_path,
        {
            "torch_version": "test",
            "cuda_version": None,
            "device_name": None,
            "surrogates": [
                {
                    "name": "tiny",
                    "pretrained": "test",
                    "weight_sha256": "weight-sha",
                    "weight_revision": "revision",
                    "device": "cpu",
                    "precision": "fp32",
                    "image_size": 4,
                    "preprocess": {},
                    "open_clip_torch_version": "test",
                    "requires_true_local_tokens": True,
                }
            ],
        },
    )
    save_json(
        evaluator_path,
        {
            "encoder_name": "CLIP-B-32",
            "model_id": "test-evaluator",
            "revision": "revision",
            "device": "cpu",
            "precision": "fp32",
            "transformers_version": "test",
            "processor_class": "test",
            "processor_use_fast": False,
            "model_class": "test",
            "weight_files": [{"sha256": "evaluator-sha", "bytes": 1}],
        },
    )
    save_json(
        root / "run_manifest.json",
        {
            "stage": "evaluated",
            "method": method,
            "source_manifest_sha256": "source-manifest-sha",
            "config_snapshot": str(config_path),
            "config_snapshot_sha256": sha256_file(config_path),
            "model_snapshot": str(model_path),
            "model_snapshot_sha256": sha256_file(model_path),
            "evaluator_snapshot": str(evaluator_path),
            "evaluator_snapshot_sha256": sha256_file(evaluator_path),
            "implementation_content_identity": "generation-implementation-sha",
            "evaluation_implementation_content_identity": "evaluation-implementation-sha",
            "samples": [
                {
                    "sample_id": "sample_01",
                    "status": "evaluated",
                    "evaluation_status": "success",
                    "source_image": "source.png",
                    "source_sha256": "source-sha",
                    "target_image": "target.png",
                    "target_sha256": "target-sha",
                    "question": "question",
                    "target_text": "target text",
                    "attack_metrics": str(attack_path),
                    "attack_metrics_sha256": sha256_file(attack_path),
                    "vector_evaluation": str(evaluation_path),
                    "vector_evaluation_sha256": sha256_file(evaluation_path),
                }
            ],
        },
    )


def test_summary_applies_predeclared_paired_thresholds(tmp_path, cpu_config) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    _write_run(
        fair,
        method="fair_pixel",
        target_gain=0.05,
        adversarial_target=0.45,
        total_seconds=10.0,
        loop_seconds=8.0,
        memory_bytes=1000,
        time_budget=None,
        checkpoint_target=0.44,
    )
    _write_run(
        frequency,
        method="frequency",
        target_gain=0.07,
        adversarial_target=0.47,
        total_seconds=7.0,
        loop_seconds=5.0,
        memory_bytes=800,
        time_budget=8.0,
        checkpoint_target=0.446,
    )
    output = tmp_path / "summary"
    one_repetition_config = replace(
        cpu_config,
        experiment=replace(cpu_config.experiment, timing_repetitions=1),
    )
    report = write_summary(
        [fair, frequency],
        output_directory=output,
        project_config=one_repetition_config,
    )
    assessment = report["predeclared_acceptance_assessment"]
    assert assessment["same_time_route"]["passed"]
    assert assessment["equivalent_effect_route"]["passed"]
    assert assessment["either_predeclared_route_passed"]
    assert (output / "results.csv").is_file()
    assert (output / "run_costs.csv").is_file()
    assert len(report["run_costs"]) == 2
    frequency_result = next(
        row for row in report["rows"] if row["method"] == "frequency"
    )
    assert frequency_result["frequency_projection"] == (
        "spatial_clip_low_frequency_reproject_with_radial_fallback"
    )
    assert frequency_result["reprojection_iterations_mean"] == 1.0
    assert frequency_result["fallback_radial_scale_mean"] == 1.0
    assert load_json(output / "summary.json")["row_count"] == 2


def test_summary_requires_configured_successful_timing_repetitions(
    tmp_path, cpu_config
) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    _write_run(
        fair,
        method="fair_pixel",
        target_gain=0.05,
        adversarial_target=0.45,
        total_seconds=10.0,
        loop_seconds=8.0,
        memory_bytes=1000,
        time_budget=None,
        checkpoint_target=0.44,
    )
    _write_run(
        frequency,
        method="frequency",
        target_gain=0.07,
        adversarial_target=0.47,
        total_seconds=7.0,
        loop_seconds=5.0,
        memory_bytes=800,
        time_budget=8.0,
        checkpoint_target=0.446,
    )
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=cpu_config,
    )
    assessment = report["predeclared_acceptance_assessment"]
    assert assessment["timing_repetitions_required"] == 3
    assert not assessment["all_paired_timing_repetitions_sufficient"]
    assert not assessment["same_time_route"]["passed"]
    assert not assessment["equivalent_effect_route"]["passed"]


def test_summary_uses_final_output_as_last_offline_reach_point(
    tmp_path, cpu_config
) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    _write_run(
        fair,
        method="fair_pixel",
        target_gain=0.05,
        adversarial_target=0.45,
        total_seconds=10.0,
        loop_seconds=8.0,
        memory_bytes=1000,
        time_budget=None,
        checkpoint_target=0.40,
    )
    _write_run(
        frequency,
        method="frequency",
        target_gain=0.049,
        adversarial_target=0.449,
        total_seconds=7.0,
        loop_seconds=5.0,
        memory_bytes=800,
        time_budget=8.0,
        checkpoint_target=0.40,
    )
    one_repetition_config = replace(
        cpu_config,
        experiment=replace(cpu_config.experiment, timing_repetitions=1),
    )
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=one_repetition_config,
    )
    pair = report["predeclared_acceptance_assessment"]["pairs"][0]
    assert pair["equivalent_effect_reached"]
    assert pair["first_reaching_checkpoint"] == "final_output"
    assert pair["first_reaching_loop_seconds"] == pytest.approx(5.0)


def test_summary_rejects_duplicate_run_directory(tmp_path, cpu_config) -> None:
    fair = tmp_path / "fair"
    _write_run(
        fair,
        method="fair_pixel",
        target_gain=0.05,
        adversarial_target=0.45,
        total_seconds=10.0,
        loop_seconds=8.0,
        memory_bytes=1000,
        time_budget=None,
        checkpoint_target=0.44,
    )
    with pytest.raises(ValueError, match="重复路径"):
        write_summary(
            [fair, fair],
            output_directory=tmp_path / "summary",
            project_config=cpu_config,
        )


def test_time_budget_uses_three_distinct_fair_run_median(tmp_path) -> None:
    runs: list[Path] = []
    for index, loop_seconds in enumerate((9.0, 7.0, 8.0), start=1):
        run = tmp_path / f"fair-{index}"
        _write_run(
            run,
            method="fair_pixel",
            target_gain=0.05,
            adversarial_target=0.45,
            total_seconds=loop_seconds + 2,
            loop_seconds=loop_seconds,
            memory_bytes=1000,
            time_budget=None,
            checkpoint_target=0.44,
        )
        runs.append(run)
    assert _time_budgets_from_runs(runs, required_repetitions=3) == {
        "sample_01": 8.0
    }
    with pytest.raises(ValueError, match="重复路径"):
        _time_budgets_from_runs([runs[0], runs[0], runs[1]], required_repetitions=3)
    model = load_json(runs[2] / "model_snapshot.json")
    model["surrogates"][0]["weight_sha256"] = "different-weight"
    save_json(runs[2] / "model_snapshot.json", model)
    with pytest.raises(ValueError, match="条件"):
        _time_budgets_from_runs(runs, required_repetitions=3)


def test_summary_rejects_cross_method_sample_identity_mismatch(
    tmp_path, cpu_config
) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    for root, method in ((fair, "fair_pixel"), (frequency, "frequency")):
        _write_run(
            root,
            method=method,
            target_gain=0.05,
            adversarial_target=0.45,
            total_seconds=10.0,
            loop_seconds=8.0,
            memory_bytes=1000,
            time_budget=8.0 if method == "frequency" else None,
            checkpoint_target=0.45,
        )
    manifest = load_json(frequency / "run_manifest.json")
    manifest["samples"][0]["target_sha256"] = "different-target"
    save_json(frequency / "run_manifest.json", manifest)
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=cpu_config,
    )
    assessment = report["predeclared_acceptance_assessment"]
    assert assessment["paired_sample_count"] == 0
    assert assessment["sample_identity_mismatch_ids"] == ["sample_01"]
    assert assessment["failed_or_missing_paired_sample_ids"] == ["sample_01"]


def test_summary_gates_acceptance_when_model_conditions_differ(
    tmp_path, cpu_config
) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    for root, method in ((fair, "fair_pixel"), (frequency, "frequency")):
        _write_run(
            root,
            method=method,
            target_gain=0.07 if method == "frequency" else 0.05,
            adversarial_target=0.47 if method == "frequency" else 0.45,
            total_seconds=7.0 if method == "frequency" else 10.0,
            loop_seconds=5.0 if method == "frequency" else 8.0,
            memory_bytes=800 if method == "frequency" else 1000,
            time_budget=8.0 if method == "frequency" else None,
            checkpoint_target=0.446,
        )
    model = load_json(frequency / "model_snapshot.json")
    model["surrogates"][0]["weight_sha256"] = "different-weight"
    save_json(frequency / "model_snapshot.json", model)
    one_repetition_config = replace(
        cpu_config,
        experiment=replace(cpu_config.experiment, timing_repetitions=1),
    )
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=one_repetition_config,
    )
    assessment = report["predeclared_acceptance_assessment"]
    assert not assessment["all_paired_run_conditions_comparable"]
    assert not assessment["same_time_route"]["passed"]
    assert not assessment["equivalent_effect_route"]["passed"]


def test_sample_missing_from_one_method_remains_in_union_denominator(
    tmp_path, cpu_config
) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    _write_run(
        fair,
        method="fair_pixel",
        target_gain=0.05,
        adversarial_target=0.45,
        total_seconds=10.0,
        loop_seconds=8.0,
        memory_bytes=1000,
        time_budget=None,
        checkpoint_target=0.44,
    )
    _write_run(
        frequency,
        method="frequency",
        target_gain=0.07,
        adversarial_target=0.47,
        total_seconds=7.0,
        loop_seconds=5.0,
        memory_bytes=800,
        time_budget=8.0,
        checkpoint_target=0.446,
    )
    fair_manifest = load_json(fair / "run_manifest.json")
    extra = dict(fair_manifest["samples"][0])
    extra["sample_id"] = "sample_02"
    extra["source_sha256"] = "second-source"
    fair_manifest["samples"].append(extra)
    save_json(fair / "run_manifest.json", fair_manifest)
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=cpu_config,
    )
    assessment = report["predeclared_acceptance_assessment"]
    assert assessment["planned_paired_sample_count"] == 2
    assert assessment["failed_or_missing_paired_sample_ids"] == ["sample_02"]
    assert assessment["equivalent_effect_route"]["reach_rate"] == 0.5


def test_summary_excludes_tampered_metric_artifact_from_success(
    tmp_path, cpu_config
) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    for root, method in ((fair, "fair_pixel"), (frequency, "frequency")):
        _write_run(
            root,
            method=method,
            target_gain=0.05,
            adversarial_target=0.45,
            total_seconds=10.0,
            loop_seconds=8.0,
            memory_bytes=1000,
            time_budget=8.0 if method == "frequency" else None,
            checkpoint_target=0.45,
        )
    evaluation_path = frequency / "sample_01" / "vector_evaluation.json"
    evaluation = load_json(evaluation_path)
    evaluation["cosine_metrics"]["target_cosine_gain"] = 1.0
    save_json(evaluation_path, evaluation)
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=cpu_config,
    )
    assessment = report["predeclared_acceptance_assessment"]
    assert assessment["paired_sample_count"] == 0
    assert assessment["failed_or_missing_paired_sample_ids"] == ["sample_01"]
    frequency_aggregate = next(
        item for item in report["aggregates"] if item["method"] == "frequency"
    )
    assert frequency_aggregate["evaluated_success_rows"] == 0


def test_summary_does_not_call_fixed_steps_equal_time_without_budget(tmp_path, cpu_config) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    _write_run(
        fair,
        method="fair_pixel",
        target_gain=0.05,
        adversarial_target=0.45,
        total_seconds=10.0,
        loop_seconds=8.0,
        memory_bytes=1000,
        time_budget=None,
        checkpoint_target=0.44,
    )
    _write_run(
        frequency,
        method="frequency",
        target_gain=0.10,
        adversarial_target=0.50,
        total_seconds=6.0,
        loop_seconds=4.0,
        memory_bytes=700,
        time_budget=None,
        checkpoint_target=0.46,
    )
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=cpu_config,
    )
    same_time = report["predeclared_acceptance_assessment"]["same_time_route"]
    assert not same_time["budgets_verified_compatible"]
    assert not same_time["passed"]


def test_failed_paired_sample_stays_in_reach_rate_denominator(tmp_path, cpu_config) -> None:
    fair = tmp_path / "fair"
    frequency = tmp_path / "frequency"
    _write_run(
        fair,
        method="fair_pixel",
        target_gain=0.05,
        adversarial_target=0.45,
        total_seconds=10.0,
        loop_seconds=8.0,
        memory_bytes=1000,
        time_budget=None,
        checkpoint_target=0.44,
    )
    _write_run(
        frequency,
        method="frequency",
        target_gain=0.07,
        adversarial_target=0.47,
        total_seconds=7.0,
        loop_seconds=5.0,
        memory_bytes=800,
        time_budget=8.0,
        checkpoint_target=0.446,
    )
    fair_manifest = load_json(fair / "run_manifest.json")
    fair_second = dict(fair_manifest["samples"][0])
    fair_second["sample_id"] = "sample_02"
    fair_manifest["samples"].append(fair_second)
    save_json(fair / "run_manifest.json", fair_manifest)
    frequency_manifest = load_json(frequency / "run_manifest.json")
    frequency_manifest["samples"].append(
        {
            "sample_id": "sample_02",
            "status": "failed",
            "evaluation_status": "failed",
            "source_sha256": "second-source",
            "target_sha256": "target-sha",
            "failure": {"type": "RuntimeError", "message": "synthetic failure"},
        }
    )
    save_json(frequency / "run_manifest.json", frequency_manifest)
    report = write_summary(
        [fair, frequency],
        output_directory=tmp_path / "summary",
        project_config=cpu_config,
    )
    assessment = report["predeclared_acceptance_assessment"]
    assert assessment["planned_paired_sample_count"] == 2
    assert assessment["paired_sample_count"] == 1
    assert assessment["failed_or_missing_paired_sample_ids"] == ["sample_02"]
    assert assessment["equivalent_effect_route"]["reach_rate"] == 0.5
    assert not assessment["either_predeclared_route_passed"]
