"""原始会话和问答读取，以及数据集审计。"""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path


from frequency_rag.image_selection.categories import CATEGORIES
from frequency_rag.common.io import load_json, save_json, sha256_file


def data_root(root):
    root = Path(root).resolve()
    for candidate in (root / "benchmark" / "data", root / "data", root):
        if (candidate / "dialog").is_dir() and (candidate / "image").is_dir():
            return candidate
    raise FileNotFoundError("未找到原始数据集的 dialog 和 image 目录。")


def _list(value):
    return value if isinstance(value, list) else ([] if value is None else [value])


def read_dialog(root, dataset):
    root = data_root(root)
    path = root / "dialog" / (dataset + ".json")
    raw = load_json(path)
    dialog = copy.deepcopy(raw)
    rounds = set()
    image_count = 0
    image_hashes = {}
    for session in dialog["multi_session_dialogues"]:
        for turn in session["dialogues"]:
            rid = str(turn["round"])
            if rid in rounds:
                raise ValueError(f"跨会话重复轮次标识，不能无歧义映射：{rid}")
            rounds.add(rid)
            images = []
            for image in _list(turn.get("input_image")):
                name = str(image).replace("\\", "/").split("/")[-1]
                candidate = root / "image" / dataset / name
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                images.append(str(candidate.resolve()))
                image_hashes[str(candidate.resolve())] = sha256_file(candidate)
            turn["input_image"] = images
            image_count += len(images)
    qas = []
    for i, qa in enumerate(dialog.get("human-annotated QAs", [])):
        item = copy.deepcopy(qa)
        item["qa_id"] = f"{dataset}:qa_{i}"
        item["clue"] = [str(x) for x in _list(item.get("clue"))]
        missing = set(item["clue"]) - rounds
        if missing:
            raise ValueError(f"问答引用不存在的轮次：{missing}")
        item["query_images"] = []
        for image in _list(item.get("question_image")):
            if not image:
                continue
            candidate = root / "image" / dataset / str(image).replace("\\", "/").split("/")[-1]
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            item["query_images"].append(str(candidate.resolve()))
        qas.append(item)
    return dialog, qas, {"dataset": dataset, "source": str(path), "sha256": sha256_file(path),
                         "sessions": len(dialog["multi_session_dialogues"]), "turns": len(rounds),
                         "images": image_count, "questions": len(qas),
                         "image_sha256": image_hashes,
                         "question_categories": dict(Counter(q.get("point", "") for q in qas))}


def prepare_benchmark(root, output, datasets=None):
    root = data_root(root)
    datasets = datasets or [p.stem for p in sorted((root / "dialog").glob("*.json"))]
    if not datasets:
        raise ValueError("原数据集目录没有对话文件。")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    audits = []
    for dataset in datasets:
        dialog, qas, audit = read_dialog(root, dataset)
        save_json(output / f"{dataset}.json", {"dialog": dialog, "qas": qas, "audit": audit})
        audits.append(audit)
    report = {"datasets": audits, "injection_categories": CATEGORIES,
              "per_category_per_dataset": 4, "planned_injections": len(datasets) * 16,
              "note": "四类是额外注入探针；原始问答类别逐项保留，不强行归并。"}
    save_json(output / "audit.json", report)
    return report
