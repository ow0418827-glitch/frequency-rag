from __future__ import annotations

from datetime import datetime, timezone
import gc
from pathlib import Path
import time
import traceback
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .attacks import AttackMethod, run_attack
from .config import ProjectConfig
from .data import (
    FrozenSample,
    load_candidates_from_memgallery,
    load_frozen_manifest,
    load_queries_from_memgallery,
)
from .evaluation import MultimodalVectorEvaluator, compute_vector_metrics
from .frequency import DCTBasisCache
from .io import load_json, save_json, save_vector, sha256_file, stable_key
from .models import OpenCLIPSurrogate, load_surrogates, resolve_device
from .objective import TargetCache
from .profiling import DeviceMemoryMonitor, PhaseClock
from .selection import select_target_for_query


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def implementation_snapshot() -> dict[str, Any]:
    """记录实际执行代码内容，供无独立版本库的运行之间核对。"""
    project_root = Path(__file__).resolve().parents[2]
    candidates = [project_root / "pyproject.toml", project_root / "tools" / "run_cli.py"]
    candidates.extend(sorted((project_root / "src" / "frequency_rag").glob("*.py")))
    files = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in candidates
        if path.is_file()
    ]
    return {
        "created_at": utc_timestamp(),
        "project_root": str(project_root),
        "files": files,
        "content_identity": stable_key(
            f"{record['path']}:{record['sha256']}" for record in files
        ),
    }


def _ensure_empty_output_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"输出路径已经被文件占用：{output}")
        if any(output.iterdir()):
            raise FileExistsError(f"输出目录已存在且非空，为避免覆盖已中止：{output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _failure_record(sample: FrozenSample, exc: BaseException) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "status": "failed",
        "source_image": str(sample.source_image),
        "source_sha256": sample.source_sha256,
        "target_image": str(sample.target_image),
        "target_sha256": sample.target_sha256,
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def generate_frozen_run(
    project_config: ProjectConfig,
    *,
    manifest_path: str | Path,
    output_directory: str | Path,
    method: str | AttackMethod,
    steps: int | None = None,
    axis_ratio: float | None = None,
    sample_ids: Iterable[str] | None = None,
    max_samples: int | None = None,
    time_budgets: Mapping[str, float] | None = None,
    time_budget_source_runs: Iterable[str | Path] | None = None,
    default_time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """生成阶段：加载一次三个代理，对冻结清单逐行攻击，不加载主评估器。"""
    generation_wall_started = time.perf_counter()
    project_config.validate()
    selected_method = AttackMethod.parse(method)
    resolved_manifest = Path(manifest_path).resolve()
    manifest_validation_started = time.perf_counter()
    source_manifest, samples = load_frozen_manifest(
        resolved_manifest,
        verify_hashes=True,
        sample_ids=sample_ids,
        max_samples=max_samples,
    )
    manifest_validation_seconds = time.perf_counter() - manifest_validation_started
    source_run_records: list[dict[str, str]] = []
    for source_run in time_budget_source_runs or ():
        source_root = Path(source_run).resolve()
        source_manifest_path = source_root / "run_manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"时间预算来源运行缺少清单：{source_manifest_path}")
        source_run_manifest = load_json(source_manifest_path)
        if source_run_manifest.get("method") != AttackMethod.FAIR_PIXEL.value:
            raise ValueError(f"时间预算来源不是公平像素运行：{source_root}")
        source_run_records.append(
            {
                "run_directory": str(source_root),
                "run_manifest": str(source_manifest_path),
                "run_manifest_sha256": sha256_file(source_manifest_path),
            }
        )
    source_run_paths = [record["run_directory"] for record in source_run_records]
    if len(set(source_run_paths)) != len(source_run_paths):
        raise ValueError("时间预算来源运行包含重复路径。")
    if source_run_records and time_budgets is None:
        raise ValueError("提供了时间预算来源运行，但没有对应的逐样本时间预算。")
    if default_time_budget_seconds is not None and (
        not np.isfinite(default_time_budget_seconds)
        or float(default_time_budget_seconds) <= 0
    ):
        raise ValueError("固定时间预算必须为有限正数。")
    if time_budgets is not None:
        invalid_budgets = {
            str(key): value
            for key, value in time_budgets.items()
            if not np.isfinite(value) or float(value) <= 0
        }
        if invalid_budgets:
            raise ValueError(f"逐样本时间预算必须为有限正数：{invalid_budgets}")
        if default_time_budget_seconds is None:
            missing_budgets = [
                sample.sample_id for sample in samples if sample.sample_id not in time_budgets
            ]
            if missing_budgets:
                raise ValueError(
                    "这些冻结样本缺少逐样本时间预算："
                    + "、".join(missing_budgets)
                )
    output = _ensure_empty_output_directory(output_directory)
    config_snapshot_path = save_json(
        output / "config_snapshot.json", project_config.to_dict()
    )
    generation_implementation = implementation_snapshot()
    implementation_snapshot_path = save_json(
        output / "implementation_snapshot.json", generation_implementation
    )
    samples_manifest_path = save_json(
        output / "samples_manifest.json",
        {
            "status": "frozen_for_execution",
            "source_manifest": str(resolved_manifest),
            "source_manifest_sha256": sha256_file(resolved_manifest) if resolved_manifest.is_file() else "",
            "source_manifest_status": source_manifest.get("status"),
            "sample_count": len(samples),
            "samples": [sample.raw for sample in samples],
        },
    )

    device = resolve_device(
        project_config.device,
        allow_fallback=project_config.runtime.allow_device_fallback,
    )
    loading_memory_monitor = DeviceMemoryMonitor(
        device,
        project_config.runtime.device_memory_sample_interval_seconds,
    )
    loading_memory_monitor.start()
    loading_clock = PhaseClock(device)
    try:
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
        model_loading_memory = loading_memory_monitor.stop()
    finally:
        loading_memory_monitor.cancel()
    model_snapshot = {
        "created_at": utc_timestamp(),
        "cold_model_loading_seconds": model_loading_seconds,
        "cold_model_loading_memory": model_loading_memory,
        "surrogates": [surrogate.snapshot() for surrogate in surrogates],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    model_snapshot_path = save_json(output / "model_snapshot.json", model_snapshot)

    target_cache = TargetCache()
    basis_cache = DCTBasisCache()
    sample_records: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        sample_output = output / sample.sample_id
        sample_output.mkdir(parents=True, exist_ok=False)
        save_json(
            sample_output / "sample_input.json",
            {
                "execution_index": index,
                "sample_id": sample.sample_id,
                "question": sample.question,
                "answer": sample.answer,
                "source_image": str(sample.source_image),
                "source_sha256": sample.source_sha256,
                "target_image": str(sample.target_image),
                "target_sha256": sample.target_sha256,
                "target_text": sample.target_text,
                "selection_score": sample.selection_score,
                "phi_r": sample.phi_r,
                "phi_contra": sample.phi_contra,
                "phi_c": sample.phi_c,
            },
        )
        budget = (
            float(time_budgets[sample.sample_id])
            if time_budgets and sample.sample_id in time_budgets
            else default_time_budget_seconds
        )
        try:
            result = run_attack(
                sample.source_image,
                sample.target_image,
                sample.target_text,
                project_config,
                method=selected_method,
                steps=steps,
                axis_ratio=axis_ratio,
                time_budget_seconds=budget,
                output_directory=sample_output,
                surrogates=surrogates,
                target_cache=(
                    target_cache
                    if project_config.shared_engineering.cache_target_features
                    else TargetCache()
                ),
                basis_cache=basis_cache,
                target_image_key=sample.target_sha256,
            )
            attack_metrics_path = sample_output / "attack_metrics.json"
            record = {
                "sample_id": sample.sample_id,
                "status": "generated",
                "source_image": str(sample.source_image),
                "source_sha256": sample.source_sha256,
                "target_image": str(sample.target_image),
                "target_sha256": sample.target_sha256,
                "question": sample.question,
                "target_text": sample.target_text,
                "output_directory": str(sample_output),
                "adversarial_image": result.metadata["output_image"],
                "adversarial_sha256": result.metadata["output_image_sha256"],
                "attack_metrics": str(attack_metrics_path),
                "attack_metrics_sha256": sha256_file(attack_metrics_path),
                "attack": result.metadata["attack"],
                "frequency": result.metadata["frequency"],
                "timing": result.metadata["timing"],
                "memory": result.metadata["memory"],
                "decoded_image_audit": result.metadata["decoded_image_audit"],
                "evaluation_status": "pending",
            }
        except Exception as exc:
            record = _failure_record(sample, exc)
            save_json(sample_output / "failure.json", record)
        sample_records.append(record)

    succeeded = sum(record["status"] == "generated" for record in sample_records)
    run_status = "complete" if succeeded == len(samples) else (
        "all_failed" if succeeded == 0 else "partial_failure"
    )
    run_manifest = {
        "schema_version": 1,
        "stage": "generation",
        "status": run_status,
        "created_at": utc_timestamp(),
        "method": selected_method.value,
        "planned_sample_count": len(samples),
        "generated_sample_count": succeeded,
        "failed_sample_count": len(samples) - succeeded,
        "source_manifest": str(resolved_manifest),
        "source_manifest_sha256": sha256_file(resolved_manifest) if resolved_manifest.is_file() else "",
        "selection_stage": {
            "mode": "pre_frozen_shared_manifest",
            "selection_model_loaded_during_generation": False,
            "manifest_and_image_hash_validation_seconds": manifest_validation_seconds,
        },
        "config_snapshot": str(config_snapshot_path),
        "config_snapshot_sha256": sha256_file(config_snapshot_path),
        "samples_manifest": str(samples_manifest_path),
        "samples_manifest_sha256": sha256_file(samples_manifest_path),
        "model_snapshot": str(model_snapshot_path),
        "model_snapshot_sha256": sha256_file(model_snapshot_path),
        "implementation_snapshot": str(implementation_snapshot_path),
        "implementation_snapshot_sha256": sha256_file(implementation_snapshot_path),
        "implementation_content_identity": generation_implementation["content_identity"],
        "cold_model_loading_seconds": model_loading_seconds,
        "cold_model_loading_memory": model_loading_memory,
        "wall_seconds_after_model_loading": time.perf_counter() - run_started,
        "generation_wall_seconds_including_manifest_validation_and_model_load": (
            time.perf_counter() - generation_wall_started
        ),
        "time_budget_provenance": {
            "mode": (
                "per_sample_reference_run_median"
                if time_budgets is not None and source_run_records
                else "per_sample_external_mapping"
                if time_budgets is not None
                else "fixed_seconds"
                if default_time_budget_seconds is not None
                else "not_used"
            ),
            "configured_required_repetitions": project_config.experiment.timing_repetitions,
            "source_runs": source_run_records,
            "per_sample_seconds": (
                {key: float(value) for key, value in sorted(time_budgets.items())}
                if time_budgets is not None
                else None
            ),
            "fixed_seconds": default_time_budget_seconds,
        },
        "target_cache": target_cache.stats(),
        "basis_cache": basis_cache.stats(),
        "primary_evaluator_loaded": False,
        "samples": sample_records,
    }
    save_json(output / "run_manifest.json", run_manifest)
    return run_manifest


def evaluate_generated_run(
    project_config: ProjectConfig,
    *,
    run_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """离线评估阶段：重新读取落盘图片，加载一次主评估器并输出四类向量。"""
    project_config.validate()
    run_root = Path(run_directory).resolve()
    evaluation_wall_started = time.perf_counter()
    manifest_path = run_root / "run_manifest.json"
    run_manifest = load_json(manifest_path)
    if run_manifest.get("stage") not in {"generation", "evaluated"}:
        raise ValueError("运行目录不是本项目的生成或已评估产物。")
    if run_manifest.get("stage") == "evaluated" and not overwrite:
        raise FileExistsError("该运行已经评估；如需重算请显式允许覆盖评估产物。")

    recorded_config_path = Path(run_manifest.get("config_snapshot", ""))
    if not recorded_config_path.is_absolute():
        recorded_config_path = run_root / recorded_config_path
    if not recorded_config_path.is_file():
        raise FileNotFoundError(f"运行缺少配置快照：{recorded_config_path}")
    expected_config_snapshot_sha256 = run_manifest.get("config_snapshot_sha256")
    if (
        not expected_config_snapshot_sha256
        or sha256_file(recorded_config_path) != expected_config_snapshot_sha256
    ):
        raise ValueError("生成阶段配置快照缺少摘要或已经改变。")
    recorded_config = load_json(recorded_config_path)
    if recorded_config != project_config.to_dict():
        raise ValueError(
            "当前评估配置与生成阶段快照不一致；请使用生成时的配置文件，"
            "避免悄悄更换评估器、设备或预处理。"
        )

    evaluation_implementation = implementation_snapshot()
    evaluation_implementation_path = save_json(
        run_root / "evaluation_implementation_snapshot.json",
        evaluation_implementation,
    )

    precision = (
        project_config.evaluation.precision_cuda
        if project_config.device.startswith("cuda")
        else project_config.evaluation.precision_cpu
    )
    device = resolve_device(
        project_config.device,
        allow_fallback=project_config.runtime.allow_device_fallback,
    )
    load_clock = PhaseClock(device)
    loading_memory_monitor = DeviceMemoryMonitor(
        device,
        project_config.runtime.device_memory_sample_interval_seconds,
    )
    loading_memory_monitor.start()
    try:
        load_clock.start()
        evaluator = MultimodalVectorEvaluator(
            project_config.evaluation.primary_encoder,
            expected_model_id=project_config.evaluation.primary_model_id,
            device=str(device),
            precision=precision,
            allow_downloads=project_config.runtime.allow_downloads,
            allow_device_fallback=project_config.runtime.allow_device_fallback,
            hash_weights=project_config.runtime.hash_model_weights,
        )
        evaluator_loading_seconds = load_clock.stop()
        evaluator_loading_memory = loading_memory_monitor.stop()
    finally:
        loading_memory_monitor.cancel()
    evaluator_snapshot = evaluator.snapshot()
    evaluator_snapshot["cold_model_loading_seconds"] = evaluator_loading_seconds
    evaluator_snapshot["cold_model_loading_memory"] = evaluator_loading_memory
    evaluator_snapshot_path = save_json(
        run_root / "evaluator_snapshot.json", evaluator_snapshot
    )

    evaluated = 0
    evaluation_failures = 0
    for record in run_manifest.get("samples", []):
        if record.get("status") not in {"generated", "evaluated"}:
            continue
        sample_output = Path(record["output_directory"])
        existing = sample_output / "vector_evaluation.json"
        if existing.exists() and not overwrite:
            raise FileExistsError(f"样本已有评估产物：{existing}")
        monitor: DeviceMemoryMonitor | None = None
        try:
            expected_attack_metrics_sha256 = record.get("attack_metrics_sha256")
            if not expected_attack_metrics_sha256:
                raise ValueError("攻击指标缺少生成阶段内容摘要。")
            actual_attack_metrics_sha256 = sha256_file(record["attack_metrics"])
            if actual_attack_metrics_sha256 != expected_attack_metrics_sha256:
                raise ValueError("攻击指标在生成与评估之间发生了改变。")
            attack_metadata = load_json(record["attack_metrics"])
            expected_adversarial_sha256 = (
                record.get("adversarial_sha256")
                or attack_metadata.get("output_image_sha256")
            )
            identities = {
                "source": {
                    "path": record["source_image"],
                    "expected_sha256": record.get("source_sha256"),
                },
                "target": {
                    "path": record["target_image"],
                    "expected_sha256": record.get("target_sha256"),
                },
                "adversarial": {
                    "path": record["adversarial_image"],
                    "expected_sha256": expected_adversarial_sha256,
                },
            }
            for name, identity in identities.items():
                if not identity["expected_sha256"]:
                    raise ValueError(f"{name} 图片缺少生成阶段内容摘要。")
                actual_sha256 = sha256_file(identity["path"])
                identity["actual_sha256"] = actual_sha256
                identity["matches"] = actual_sha256 == identity["expected_sha256"]
                if not identity["matches"]:
                    raise ValueError(f"{name} 图片在生成与评估之间发生了改变。")
            monitor = DeviceMemoryMonitor(
                evaluator.device,
                project_config.runtime.device_memory_sample_interval_seconds,
            )
            monitor.start()
            clock = PhaseClock(evaluator.device)
            clock.start()
            source_vector = evaluator.encode_image(record["source_image"])
            adversarial_vector = evaluator.encode_image(record["adversarial_image"])
            target_vector = evaluator.encode_image(record["target_image"])
            query_vector = evaluator.encode_text(record["question"])
            metrics = compute_vector_metrics(
                sample_id=record["sample_id"],
                encoder_name=evaluator.encoder_name,
                encoder_model_id=evaluator.model_id,
                query_text=record["question"],
                source_vector=source_vector,
                adversarial_vector=adversarial_vector,
                target_vector=target_vector,
                query_vector=query_vector,
            )
            metrics["image_identity"] = identities
            if metrics["embedding_dimension"] != project_config.evaluation.embedding_dimension:
                raise RuntimeError(
                    "主评估向量维度不一致："
                    f"得到 {metrics['embedding_dimension']}，"
                    f"要求 {project_config.evaluation.embedding_dimension}。"
                )
            save_vector(sample_output / "source_vector.npy", source_vector)
            save_vector(sample_output / "adversarial_vector.npy", adversarial_vector)
            save_vector(sample_output / "target_vector.npy", target_vector)
            save_vector(sample_output / "query_vector.npy", query_vector)

            checkpoint_metrics: list[dict[str, Any]] = []
            for checkpoint in attack_metadata.get("checkpoints", []):
                if not checkpoint.get("image"):
                    continue
                if not checkpoint.get("image_sha256"):
                    raise ValueError(
                        f"检查点 {checkpoint.get('label')} 缺少生成阶段内容摘要。"
                    )
                checkpoint_actual_sha256 = sha256_file(checkpoint["image"])
                if checkpoint_actual_sha256 != checkpoint["image_sha256"]:
                    raise ValueError(
                        f"检查点 {checkpoint.get('label')} 在生成与评估之间发生了改变。"
                    )
                checkpoint_vector = evaluator.encode_image(checkpoint["image"])
                checkpoint_result = compute_vector_metrics(
                    sample_id=record["sample_id"],
                    encoder_name=evaluator.encoder_name,
                    encoder_model_id=evaluator.model_id,
                    query_text=record["question"],
                    source_vector=source_vector,
                    adversarial_vector=checkpoint_vector,
                    target_vector=target_vector,
                    query_vector=query_vector,
                )
                checkpoint_record = {
                    **{key: value for key, value in checkpoint.items() if key != "decoded_audit"},
                    "actual_image_sha256": checkpoint_actual_sha256,
                    "image_sha256_matches": True,
                    "vector_metrics": checkpoint_result,
                }
                checkpoint_metrics.append(checkpoint_record)
                save_json(
                    sample_output / "checkpoints" / f"{checkpoint['label']}_evaluation.json",
                    checkpoint_record,
                )
            metrics["checkpoints"] = checkpoint_metrics
            evaluation_seconds = clock.stop()
            evaluation_memory = monitor.stop()
            metrics["cost"] = {
                "evaluation_seconds": evaluation_seconds,
                "peak_memory": evaluation_memory,
                "cold_evaluator_loading_seconds_shared_by_run": evaluator_loading_seconds,
            }
            save_json(existing, metrics)
            record["status"] = "evaluated"
            record["evaluation_status"] = "success"
            record["vector_evaluation"] = str(existing)
            record["vector_evaluation_sha256"] = sha256_file(existing)
            record["evaluation"] = metrics
            evaluated += 1
        except Exception as exc:
            record["evaluation_status"] = "failed"
            record["evaluation_failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            save_json(sample_output / "evaluation_failure.json", record["evaluation_failure"])
            evaluation_failures += 1
        finally:
            if monitor is not None:
                monitor.cancel()
            gc.collect()

    run_manifest["stage"] = "evaluated"
    run_manifest["evaluated_at"] = utc_timestamp()
    generation_failures = sum(
        record.get("status") == "failed" for record in run_manifest.get("samples", [])
    )
    planned_samples = int(
        run_manifest.get("planned_sample_count") or len(run_manifest.get("samples", []))
    )
    run_manifest["evaluation_status"] = (
        "complete"
        if evaluated == planned_samples and evaluation_failures == 0 and generation_failures == 0
        else "all_failed"
        if evaluated == 0
        else "partial_failure"
    )
    run_manifest["evaluated_sample_count"] = evaluated
    run_manifest["evaluation_failed_sample_count"] = evaluation_failures
    run_manifest["evaluation_skipped_generation_failure_count"] = generation_failures
    run_manifest["evaluator_snapshot"] = str(evaluator_snapshot_path)
    run_manifest["evaluator_snapshot_sha256"] = sha256_file(evaluator_snapshot_path)
    run_manifest["evaluation_implementation_snapshot"] = str(
        evaluation_implementation_path
    )
    run_manifest["evaluation_implementation_snapshot_sha256"] = sha256_file(
        evaluation_implementation_path
    )
    run_manifest["evaluation_implementation_content_identity"] = (
        evaluation_implementation["content_identity"]
    )
    run_manifest["cold_evaluator_loading_seconds"] = evaluator_loading_seconds
    run_manifest["cold_evaluator_loading_memory"] = evaluator_loading_memory
    run_manifest["evaluation_wall_seconds_including_model_load"] = (
        time.perf_counter() - evaluation_wall_started
    )
    generation_wall = run_manifest.get(
        "generation_wall_seconds_including_manifest_validation_and_model_load"
    )
    if generation_wall is None:
        generation_wall = (
            float(run_manifest.get("cold_model_loading_seconds") or 0)
            + float(run_manifest.get("wall_seconds_after_model_loading") or 0)
            + float(_nested_selection_validation_seconds(run_manifest))
        )
    run_manifest["measured_pipeline_stage_seconds_excluding_idle_between_commands"] = (
        float(generation_wall)
        + float(run_manifest["evaluation_wall_seconds_including_model_load"])
    )
    save_json(manifest_path, run_manifest)
    return run_manifest


def _nested_selection_validation_seconds(run_manifest: dict[str, Any]) -> float:
    selection = run_manifest.get("selection_stage", {})
    return float(selection.get("manifest_and_image_hash_validation_seconds") or 0)


def recompute_reference_selection(
    project_config: ProjectConfig,
    *,
    reference_manifest_path: str | Path,
    output_path: str | Path,
    maximum_rows: int = 10,
) -> dict[str, Any]:
    """重新执行原选图规则并逐行对比冻结结果；此阶段不加载攻击或评估模型。"""
    selection_started = time.perf_counter()
    project_config.validate()
    reference, frozen = load_frozen_manifest(reference_manifest_path, verify_hashes=True)
    candidates = load_candidates_from_memgallery(
        project_config.memgallery_root,
        project_config.selection.candidate_domains,
    )
    queries = load_queries_from_memgallery(project_config.memgallery_root, project_config.dataset)
    feature_model = OpenCLIPSurrogate(
        project_config.surrogates[0],
        device=project_config.device,
        precision=project_config.attack_precision,
        allow_downloads=project_config.runtime.allow_downloads,
        allow_device_fallback=project_config.runtime.allow_device_fallback,
        require_true_local_tokens=project_config.runtime.require_true_local_tokens,
        hash_weight=project_config.runtime.hash_model_weights,
    )
    image_vectors: list[np.ndarray] = []
    caption_vectors: list[np.ndarray] = []
    from .io import load_image, pil_to_tensor

    with torch.no_grad():
        for candidate in candidates:
            image = pil_to_tensor(load_image(candidate.image_path), device=feature_model.device)
            image_vectors.append(
                feature_model.encode_image(image).global_embedding[0].detach().cpu().numpy()
            )
            caption_vectors.append(
                feature_model.encode_text(candidate.caption)[0].detach().cpu().numpy()
            )
    image_array = np.stack(image_vectors)
    caption_array = np.stack(caption_vectors)
    rows: list[dict[str, Any]] = []
    for query in queries:
        with torch.no_grad():
            question_vector = feature_model.encode_text(query.question)[0].detach().cpu().numpy()
            answer_vector = feature_model.encode_text(query.answer)[0].detach().cpu().numpy()
        result = select_target_for_query(
            query,
            candidates,
            image_array,
            caption_array,
            question_vector,
            answer_vector,
            project_config.selection,
        )
        if result is None:
            continue
        rows.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "answer": query.answer,
                "source_image": str(query.source_image_path),
                "source_sha256": sha256_file(query.source_image_path),
                "target_image": str(result.target_candidate.image_path),
                "target_sha256": sha256_file(result.target_candidate.image_path),
                "target_text": result.target_text,
                "selection_score": result.score,
                "phi_r": result.phi_r,
                "phi_contra": result.phi_contra,
                "phi_c": result.phi_c,
                "selected_candidate_passed_lexical_gate": result.selected_candidate_passed_lexical_gate,
                "all_candidates_failed_lexical_gate": result.all_candidates_failed_lexical_gate,
            }
        )
        if len(rows) >= maximum_rows:
            break

    comparisons: list[dict[str, Any]] = []
    for index in range(max(len(rows), min(maximum_rows, len(frozen)))):
        actual = rows[index] if index < len(rows) else None
        expected = frozen[index] if index < len(frozen) else None
        matches = bool(
            actual
            and expected
            and actual["source_sha256"] == expected.source_sha256
            and actual["target_sha256"] == expected.target_sha256
            and actual["question"] == expected.question
            and actual["target_text"] == expected.target_text
            and abs(actual["selection_score"] - expected.selection_score) <= 1e-6
        )
        comparisons.append(
            {
                "index": index + 1,
                "matches_frozen": matches,
                "actual": actual,
                "expected_sample_id": expected.sample_id if expected else None,
            }
        )
    report = {
        "status": "match" if comparisons and all(row["matches_frozen"] for row in comparisons) else "mismatch",
        "created_at": utc_timestamp(),
        "reference_manifest": str(Path(reference_manifest_path).resolve()),
        "reference_commit": reference.get("reference_commit"),
        "candidate_count": len(candidates),
        "query_count": len(queries),
        "rows_recomputed": len(rows),
        "model_snapshot": feature_model.snapshot(),
        "implementation_snapshot": implementation_snapshot(),
        "wall_seconds_before_report_write": time.perf_counter() - selection_started,
        "comparisons": comparisons,
    }
    save_json(output_path, report)
    return report
