"""原始数据、图像对、计划生成和记忆评估的命令分派。"""
from __future__ import annotations

import json
from pathlib import Path

from frequency_rag.common.io import load_json
from frequency_rag.image_selection.dataset import prepare_benchmark
from frequency_rag.image_selection.plans import build_injection_plan
from frequency_rag.experiments.memory_runner import run_memory_experiment
from frequency_rag.image_selection.plans import build_poisoning_plan
from frequency_rag.image_selection.candidates import select_injection_pairs
from frequency_rag.memory_agent.llm import ChatModel
from frequency_rag.memory_agent.encoder import MemoryEncoder


def add_memory_commands(subparsers):
    prepare = subparsers.add_parser("memory-prepare", help="读取原始数据并审计所有会话和问答")
    prepare.add_argument("--memgallery-root", type=Path, required=True)
    prepare.add_argument("--datasets", nargs="+")
    prepare.add_argument("--output-dir", type=Path, required=True)
    pool = subparsers.add_parser("memory-select-pairs", help="候选池检索、视觉验证并生成四类问答")
    pool.add_argument("--pool", type=Path, required=True)
    pool.add_argument("--memory-config", type=Path, required=True)
    pool.add_argument("--per-category", type=int, default=4)
    pool.add_argument("--candidate-limit", type=int, default=100)
    pool.add_argument("--output", type=Path, required=True)
    plan = subparsers.add_parser("memory-plan", help="构建四类注入或原问答投毒计划")
    plan.add_argument("--bundle", type=Path, required=True)
    plan.add_argument("--family", choices=["injection", "poisoning"], required=True)
    plan.add_argument("--pairs", type=Path, required=True, help="注入图像对或投毒冻结清单")
    plan.add_argument("--per-category", type=int, default=4)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("memory-run", help="运行长期记忆智能体及分条件问答评估")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--plan", type=Path)
    run.add_argument("--memory-config", type=Path, required=True)
    run.add_argument("--attack-run", type=Path)
    run.add_argument("--conditions", nargs="+", default=["clean", "oracle", "adversarial"],
                     choices=["clean", "oracle", "adversarial", "source_injection_control"])
    run.add_argument("--output-dir", type=Path, required=True)


def dispatch_memory(args):
    if args.command == "memory-prepare":
        result = prepare_benchmark(args.memgallery_root, args.output_dir, args.datasets)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "memory-select-pairs":
        config = load_json(args.memory_config)
        result = select_injection_pairs(args.pool, args.output, MemoryEncoder(config["encoder"]),
                                        ChatModel(config["model"]), per_category=args.per_category,
                                        candidate_limit=args.candidate_limit)
        print(f"已保存 {len(result['pairs'])} 组通过视觉验证的图像对。")
    elif args.command == "memory-plan":
        if args.family == "injection":
            result = build_injection_plan(args.bundle, args.pairs, args.output,
                                          per_category=args.per_category, seed=args.seed)
        else:
            result = build_poisoning_plan(args.bundle, args.pairs, args.output)
        print(f"已保存 {len(result['samples'])} 组实验计划。")
    elif args.command == "memory-run":
        result = run_memory_experiment(args.bundle, args.plan, args.memory_config, args.output_dir,
                                       run_dir=args.attack_run, conditions=args.conditions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "complete" else 2
    return 0
