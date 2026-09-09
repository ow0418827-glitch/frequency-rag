from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

from .attacks import AttackMethod
from .benchmark import generation_conditions_signature, write_summary
from .config import ProjectConfig, load_config
from .data import load_frozen_manifest
from .io import load_json, save_json, sha256_file
from .models import load_surrogates, resolve_hf_cached_weight
from .pipeline import (
    evaluate_generated_run,
    generate_frozen_run,
    recompute_reference_selection,
)


def _default_manifest(config: ProjectConfig) -> Path:
    return config.resolve_project_path(config.reference_manifest)


def _split_sample_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result or None


def _time_budgets_from_runs(
    paths: list[Path], *, required_repetitions: int, expected_steps: int = 1000
) -> dict[str, float]:
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("时间预算参考目录包含重复路径。")
    required = max(1, int(required_repetitions))
    if len(resolved_paths) < required:
        raise ValueError(
            f"时间预算至少需要 {required} 次不同的公平像素运行，当前只有 {len(resolved_paths)} 次。"
        )
    grouped: dict[str, list[float]] = {}
    planned_ids: set[str] = set()
    condition_signatures: set[str] = set()
    for root in resolved_paths:
        manifest = load_json(root / "run_manifest.json")
        if manifest.get("method") != AttackMethod.FAIR_PIXEL.value:
            raise ValueError(f"时间预算参考必须是公平像素组：{root}")
        condition_signatures.add(generation_conditions_signature(root, manifest))
        ids_in_run: set[str] = set()
        for sample in manifest.get("samples", []):
            sample_id = str(sample.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"公平像素运行包含空样本标识：{root}")
            if sample_id in ids_in_run:
                raise ValueError(f"公平像素运行包含重复样本标识 {sample_id}：{root}")
            ids_in_run.add(sample_id)
            planned_ids.add(sample_id)
            if sample.get("status") not in {"generated", "evaluated"}:
                continue
            metrics_path = sample.get("attack_metrics")
            resolved_metrics_path = Path(metrics_path) if metrics_path else None
            if resolved_metrics_path is not None and not resolved_metrics_path.is_absolute():
                resolved_metrics_path = root / resolved_metrics_path
            if resolved_metrics_path is None or not resolved_metrics_path.is_file():
                continue
            expected_metrics_sha256 = sample.get("attack_metrics_sha256")
            if (
                not expected_metrics_sha256
                or sha256_file(resolved_metrics_path) != expected_metrics_sha256
            ):
                raise ValueError(
                    f"公平像素参考样本 {sample_id} 的攻击指标缺少摘要或已经改变。"
                )
            metrics = load_json(resolved_metrics_path)
            attack = metrics.get("attack", {})
            if (
                int(attack.get("steps_requested", -1)) != int(expected_steps)
                or int(attack.get("steps_completed", -1)) != int(expected_steps)
                or attack.get("time_budget_seconds") is not None
            ):
                raise ValueError(
                    f"公平像素参考样本 {sample_id} 不是完整的 {expected_steps} 步无时间截断运行。"
                )
            seconds = metrics.get("timing", {}).get("optimization_loop_seconds")
            if seconds is not None and math.isfinite(float(seconds)) and float(seconds) > 0:
                grouped.setdefault(sample_id, []).append(float(seconds))
    if len(condition_signatures) != 1:
        raise ValueError("公平像素参考运行的配置、模型、硬件、数据或源码条件不一致。")
    if not grouped:
        raise ValueError("参考运行中没有可用的逐样本优化循环时间。")
    insufficient = {
        sample_id: len(grouped.get(sample_id, []))
        for sample_id in sorted(planned_ids)
        if len(grouped.get(sample_id, [])) < required
    }
    if insufficient:
        details = "、".join(f"{sample_id}={count}" for sample_id, count in insufficient.items())
        raise ValueError(
            f"部分样本没有达到 {required} 次成功公平像素计时：{details}。"
        )
    return {sample_id: statistics.median(values) for sample_id, values in grouped.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frequency-rag",
        description="频域参数化图像向量扰动与公平像素对照实验框架",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="可执行配置文件路径",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="核对环境、数据和本地权重")
    doctor.add_argument("--load-models", action="store_true", help="实际加载三个代理并检查局部特征接口")

    verify = subparsers.add_parser("verify", help="验证配置与冻结样本图片的内容校验值")
    verify.add_argument("--manifest", type=Path)
    verify.add_argument("--sample-ids", nargs="*")

    select = subparsers.add_parser("select", help="冻结现有清单子集或重新核对选图一致性")
    select.add_argument("--manifest", type=Path)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--maximum-rows", type=int, default=10)
    select.add_argument("--recompute", action="store_true")
    select.add_argument("--sample-ids", nargs="*")

    attack = subparsers.add_parser("attack", help="只运行生成阶段，不加载主评估器")
    attack.add_argument(
        "--method",
        choices=[item.value for item in AttackMethod],
        required=True,
    )
    attack.add_argument("--manifest", type=Path)
    attack.add_argument("--output-dir", type=Path, required=True)
    attack.add_argument("--steps", type=int)
    attack.add_argument("--axis-ratio", type=float)
    attack.add_argument("--sample-ids", nargs="*")
    budget_group = attack.add_mutually_exclusive_group()
    budget_group.add_argument("--time-budget-seconds", type=float)
    budget_group.add_argument(
        "--time-budget-from-fair-runs",
        type=Path,
        nargs="+",
        help="从一次或多次公平像素运行提取逐样本循环时间中位数",
    )

    evaluate = subparsers.add_parser("evaluate", help="离线读取落盘图片并计算主评估向量")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")

    summarize = subparsers.add_parser("summarize", help="汇总运行并按预设验收线配对比较")
    summarize.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    return parser


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _doctor(config: ProjectConfig, *, load_models: bool) -> int:
    try:
        versions = {
            package: importlib.metadata.version(package)
            for package in ("torch", "numpy", "Pillow", "open-clip-torch", "transformers")
        }
    except importlib.metadata.PackageNotFoundError as exc:
        print(f"依赖缺失：{exc.name}")
        return 1
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "packages": versions,
        "configured_device": config.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "memgallery_root": config.memgallery_root,
        "memgallery_exists": Path(config.memgallery_root).is_dir(),
        "reference_manifest": str(_default_manifest(config)),
        "reference_manifest_exists": _default_manifest(config).is_file(),
        "surrogate_weights": [],
    }
    missing_weights = False
    for surrogate in config.surrogates:
        weight = Path(surrogate.weight_path).resolve() if surrogate.weight_path else resolve_hf_cached_weight(surrogate.hf_repo)
        exists = bool(weight and weight.is_file())
        missing_weights = missing_weights or not exists
        report["surrogate_weights"].append(
            {
                "name": surrogate.name,
                "pretrained": surrogate.pretrained,
                "hf_repo": surrogate.hf_repo,
                "path": str(weight) if weight else None,
                "exists": exists,
            }
        )
    if load_models:
        models = load_surrogates(
            config.surrogates,
            device=config.device,
            precision=config.attack_precision,
            allow_downloads=config.runtime.allow_downloads,
            allow_device_fallback=config.runtime.allow_device_fallback,
            require_true_local_tokens=config.runtime.require_true_local_tokens,
            hash_weights=config.runtime.hash_model_weights,
        )
        report["loaded_model_snapshots"] = [model.snapshot() for model in models]
        probe = torch.zeros(1, 3, 19, 23, device=models[0].device, dtype=torch.float32)
        local_shapes: list[dict[str, Any]] = []
        with torch.no_grad():
            for model in models:
                features = model.encode_image(probe)
                local_shapes.append(
                    {
                        "name": model.config.name,
                        "pretrained": model.config.pretrained,
                        "global_shape": list(features.global_embedding.shape),
                        "local_shape": list(features.patch_tokens.shape),
                        "true_local_tokens": features.patch_tokens.shape[1] > 1,
                    }
                )
        report["local_feature_interface_probe"] = local_shapes
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required_ok = bool(
        report["memgallery_exists"]
        and report["reference_manifest_exists"]
        and (not missing_weights or config.runtime.allow_downloads)
        and (torch.cuda.is_available() or config.device == "cpu" or config.runtime.allow_device_fallback)
    )
    return 0 if required_ok else 1


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            return _doctor(config, load_models=args.load_models)

        if args.command == "verify":
            manifest_path = args.manifest or _default_manifest(config)
            _, samples = load_frozen_manifest(
                manifest_path,
                verify_hashes=True,
                sample_ids=_split_sample_ids(args.sample_ids),
            )
            print(f"核查通过：{len(samples)} 行冻结样本的源图与目标图内容校验值一致。")
            return 0

        if args.command == "select":
            manifest_path = args.manifest or _default_manifest(config)
            if args.maximum_rows <= 0:
                raise ValueError("选图或冻结清单的最大行数必须为正整数。")
            if args.recompute:
                if args.sample_ids:
                    raise ValueError("重新选图按参考顺序核对，不能同时传入 sample-ids。")
                report = recompute_reference_selection(
                    config,
                    reference_manifest_path=manifest_path,
                    output_path=args.output,
                    maximum_rows=args.maximum_rows,
                )
                print(f"重新选图核对状态：{report['status']}；报告：{Path(args.output).resolve()}")
                return 0 if report["status"] == "match" else 2
            source, samples = load_frozen_manifest(
                manifest_path,
                verify_hashes=True,
                sample_ids=_split_sample_ids(args.sample_ids),
            )
            selected = samples[: args.maximum_rows]
            save_json(
                args.output,
                {
                    "status": "frozen_for_execution",
                    "source_manifest": str(Path(manifest_path).resolve()),
                    "source_manifest_sha256": sha256_file(manifest_path),
                    "reference_commit": source.get("reference_commit"),
                    "num_samples": len(selected),
                    "samples": [sample.raw for sample in selected],
                },
            )
            print(f"已冻结 {len(selected)} 行执行清单：{Path(args.output).resolve()}")
            return 0

        if args.command == "attack":
            manifest_path = args.manifest or _default_manifest(config)
            if (
                args.time_budget_from_fair_runs
                and args.method
                not in {
                    AttackMethod.FREQUENCY.value,
                    AttackMethod.PROGRESSIVE_FREQUENCY.value,
                }
            ):
                raise ValueError("公平像素参考时间预算只用于固定频带或渐进扩频方法。")
            budgets = (
                _time_budgets_from_runs(
                    args.time_budget_from_fair_runs,
                    required_repetitions=config.experiment.timing_repetitions,
                    expected_steps=config.attack.reference_steps,
                )
                if args.time_budget_from_fair_runs
                else None
            )
            result = generate_frozen_run(
                config,
                manifest_path=manifest_path,
                output_directory=args.output_dir,
                method=args.method,
                steps=args.steps,
                axis_ratio=args.axis_ratio,
                sample_ids=_split_sample_ids(args.sample_ids),
                time_budgets=budgets,
                time_budget_source_runs=args.time_budget_from_fair_runs,
                default_time_budget_seconds=args.time_budget_seconds,
            )
            print(
                f"生成阶段完成：成功 {result['generated_sample_count']}/"
                f"{result['planned_sample_count']}；运行清单："
                f"{Path(args.output_dir).resolve() / 'run_manifest.json'}"
            )
            return 0 if result["status"] == "complete" else 2

        if args.command == "evaluate":
            result = evaluate_generated_run(
                config,
                run_directory=args.run_dir,
                overwrite=args.overwrite,
            )
            print(
                f"离线评估完成：成功 {result['evaluated_sample_count']}；失败 "
                f"{result['evaluation_failed_sample_count']}。"
            )
            return 0 if result["evaluation_status"] == "complete" else 2

        if args.command == "summarize":
            result = write_summary(
                args.run_dirs,
                output_directory=args.output_dir,
                project_config=config,
            )
            print(result["conclusion"])
            print(f"汇总文件：{Path(args.output_dir).resolve() / 'summary.json'}")
            return 0
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
