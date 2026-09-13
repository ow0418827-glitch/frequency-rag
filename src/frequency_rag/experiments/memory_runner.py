"""记忆写入、探针回答、独立裁判及按条件／类别的统计。"""
from __future__ import annotations

from dataclasses import asdict
from frequency_rag.common.paths import PACKAGE_ROOT
from pathlib import Path


from frequency_rag.common.io import load_json, save_json, sha256_file
from frequency_rag.memory_agent.backends import make_memory, available_backends
from frequency_rag.memory_agent.scenarios import adversarial_images, condition_dialog
from frequency_rag.memory_agent.agent import MemoryAgent
from frequency_rag.evaluation.judge import judge_answer
from frequency_rag.evaluation.memory_metrics import aggregate, token_f1, retrieval_metrics
from frequency_rag.memory_agent.llm import ChatModel
from frequency_rag.memory_agent.encoder import MemoryEncoder


def validate_plan(bundle, plan):
    if plan.get("dataset") != bundle["audit"]["dataset"]:
        raise ValueError("实验计划与数据集不匹配。")
    if plan.get("family") not in {"injection", "poisoning"} or not plan.get("samples"):
        raise ValueError("实验计划必须包含有效类型和非空样本。")
    for field in ("sample_id", "turn_id", "qa_id"):
        ids = [s[field] for s in plan["samples"]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"计划有重复的 {field}。")
    for sample in plan["samples"]:
        for key in ("source", "target"):
            if sha256_file(sample[key + "_image"]) != sample[key + "_sha256"]:
                raise ValueError("计划图像内容摘要不匹配。")


def run_memory_experiment(bundle_path, plan_path, config_path, output, *, run_dir=None,
                          conditions=("clean", "oracle", "adversarial"), model=None, judge=None,
                          encoder=None, memory_factory=None):
    bundle, config = load_json(bundle_path), load_json(config_path)
    def reject_secrets(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in {"api_key", "authorization", "password", "token"} and child:
                    raise ValueError("记忆配置禁止直接嵌入密钥，请使用环境变量。")
                reject_secrets(child)
        elif isinstance(value, list):
            for child in value:
                reject_secrets(child)
    reject_secrets(config)
    plan = load_json(plan_path) if plan_path else {"samples": [], "family": "none"}
    if len(set(conditions)) != len(conditions) or not conditions:
        raise ValueError("实验条件不能重复或为空。")
    if set(conditions) - {"clean", "oracle", "adversarial", "source_injection_control"}:
        raise ValueError("未知实验条件。")
    if config["backend"] not in available_backends():
        raise ValueError("未知记忆后端。")
    if not 0 < config.get("top_k", 3) <= config.get("candidates", 10):
        raise ValueError("检索数量不合法。")
    if Path(output).exists():
        raise FileExistsError(output)
    for path, expected in bundle["audit"].get("image_sha256", {}).items():
        if sha256_file(path) != expected:
            raise ValueError("原数据图片在数据包准备后发生改变，请重新审计。")
    if plan_path:
        validate_plan(bundle, plan)
    elif list(conditions) != ["clean"]:
        raise ValueError("没有实验计划只能运行干净条件。")
    images = None
    if "adversarial" in conditions:
        if not run_dir:
            raise ValueError("对抗条件必须提供现有扰动运行目录。")
        images = adversarial_images(run_dir, plan["samples"], config.get("epsilon", 16 / 255))
    model = model or ChatModel(config["model"])
    judge = judge or ChatModel(config.get("judge", config["model"]))
    encoder = encoder or (MemoryEncoder(config["encoder"]) if config["backend"] != "mem0" else None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    rows = []
    provenance = {"bundle_sha256": sha256_file(bundle_path),
                  "plan_sha256": sha256_file(plan_path) if plan_path else None,
                  "config_sha256": sha256_file(config_path),
                  "backend": config["backend"], "implementation": "independent_paper_based",
                  "encoder": encoder.snapshot() if hasattr(encoder, "snapshot") else {"kind": "injected_test_double_or_mem0"},
                  "conditions": list(conditions), "seed": config.get("seed", 42),
                  "model": config["model"], "source_code": {p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p) for p in sorted(PACKAGE_ROOT.rglob("*.py"))}}
    if run_dir:
        attack_manifest_path = Path(run_dir) / "run_manifest.json"
        attack_manifest = load_json(attack_manifest_path)
        provenance["attack_run"] = {"directory": str(Path(run_dir).resolve()),
                                    "manifest_sha256": sha256_file(attack_manifest_path),
                                    "method": attack_manifest.get("method"),
                                    "adversarial_images": {s["sample_id"]: sha256_file(images[s["sample_id"]])
                                                           for s in plan["samples"]} if images else {}}
    save_json(output / "manifest.json", provenance)
    for condition in conditions:
        directory = output / condition
        directory.mkdir()
        memory = (memory_factory or make_memory)(config, encoder, model)
        agent = MemoryAgent(memory, model)
        try:
            dialog = condition_dialog(bundle, plan, condition, images=images, describer=model)
            save_json(directory / "dialog.json", dialog)
            agent.ingest(dialog)
            memory.save(directory / "memory.json")
        except Exception as exc:
            save_json(directory / "failure.json", {"phase": "ingestion", "error_type": type(exc).__name__})
            save_json(output / "status.json", {"status": "failed", "condition": condition, "phase": "ingestion"})
            raise
        qas = [{**q, "probe_type": "original"} for q in bundle["qas"]]
        for sample in plan["samples"]:
            if condition == "clean" and plan["family"] == "injection":
                # 原论文干净条件没有新增轮次；不能拿尚未上传的源图答案给干净模型判错。
                continue
            qas.append({"qa_id": "attack:" + sample["qa_id"], "question": sample["question"],
                        "answer": sample["answer"], "target_answer": sample["target_answer"],
                        "point": sample.get("category", "poisoning"), "clue": [sample["turn_id"]],
                        "attack_turn_id": sample["turn_id"], "probe_type": "attack", "query_images": []})
        for qa in qas:
            row = {"qa_id": qa["qa_id"], "condition": condition, "category": qa.get("point", ""),
                   "probe_type": qa["probe_type"], "status": "ok"}
            try:
                answer, recalled = agent.answer(qa["question"], qa.get("query_images", []))
                result = judge_answer(judge, qa, answer)
                row.update(answer=answer, reference_answer=qa["answer"], question=qa["question"],
                           correct=result["correct"], judge_reason=result.get("reason", ""),
                           target_claim_present=result["target_claim_present"] if qa.get("target_answer") else None,
                           f1=token_f1(answer, qa["answer"]), retrieval=asdict(recalled),
                           attack_retrieved=qa.get("attack_turn_id") in {e.entry_id for e in recalled.entries},
                           **retrieval_metrics(recalled.entries, qa.get("clue", [])))
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__)
            rows.append(row)
            save_json(directory / "answers.json", [r for r in rows if r["condition"] == condition])
        # 问答不回写记忆，避免先前探针污染后续回答。
    summary = aggregate(rows)
    save_json(output / "summary.json", summary)
    save_json(output / "status.json", {"status": summary["status"]})
    return summary
