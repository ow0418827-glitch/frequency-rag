"""将验证后的图像对绑定到会话，生成注入或投毒计划。"""
from __future__ import annotations
import copy
from frequency_rag.image_selection.data import load_frozen_manifest
from pathlib import Path
import random
from frequency_rag.common.io import load_json, save_json, sha256_file

from frequency_rag.image_selection.categories import CATEGORIES

def build_injection_plan(bundle, pairs_path, output, *, per_category=4, seed=42):
    """pairs 是经视觉确认的源／目标对；不以关键词命中代替语义验证。"""
    if per_category < 1:
        raise ValueError("每类数量必须为正。")
    bundle = load_json(bundle)
    source = load_json(pairs_path)
    pairs = source["pairs"]
    sessions = bundle["dialog"]["multi_session_dialogues"]
    if len(sessions) < 2:
        raise ValueError("至少需要两个会话，以避免在最后会话插入。")
    rng = random.Random(seed)
    selected = []
    for category in CATEGORIES:
        eligible = [p for p in pairs if p["category"] == category and p.get("validated") is True]
        if len(eligible) < per_category:
            raise ValueError(f"{category} 缺少已验证图像对：需要 {per_category}，只有 {len(eligible)}。")
        for pair in rng.sample(eligible, per_category):
            item = copy.deepcopy(pair)
            for key in ("source_image", "target_image"):
                path = Path(item[key])
                if not path.is_absolute():
                    path = Path(pairs_path).resolve().parent / path
                item[key] = str(path.resolve())
                item[key.replace("image", "sha256")] = sha256_file(path)
            if item["source_sha256"] == item["target_sha256"]:
                raise ValueError("源图与目标图相同。")
            for key in ("question", "answer", "target_answer", "target_text", "user"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"图像对缺少 {key}。")
            if item["answer"].strip().casefold() == item["target_answer"].strip().casefold():
                raise ValueError("真实答案与目标答案不能相同。")
            index = len(selected)
            item.update(sample_id=f"inject_{index:03d}", turn_id=f"INJECT:{index:03d}",
                        session_id=str(rng.choice(sessions[:-1])["session_id"]),
                        qa_id=f"inject:qa_{index:03d}", family="injection")
            selected.append(item)
    result = {"family": "injection", "dataset": bundle["audit"]["dataset"], "seed": seed,
              "pool_provenance": source.get("provenance", "unspecified"),
              "pairs_sha256": sha256_file(pairs_path), "samples": selected}
    save_json(output, result)
    # 同一组图像直接交给现有 attack 命令，不重新选图。
    save_json(Path(output).with_suffix(".attack.json"), {"status": "frozen_for_execution", "samples": selected})
    return result


def build_poisoning_plan(bundle_path, manifest_path, output):
    bundle = load_json(bundle_path)
    _, samples = load_frozen_manifest(manifest_path)
    turns = {str(t["round"]): (str(s["session_id"]), t) for s in bundle["dialog"]["multi_session_dialogues"]
             for t in s["dialogues"]}
    selected = []
    for sample in samples:
        matches = [q for q in bundle["qas"] if q["question"] == sample.question]
        if len(matches) != 1:
            raise ValueError(f"{sample.sample_id} 的问题未唯一匹配原数据集；不能猜测投毒轮次。")
        qa = matches[0]
        targets = [(sid, turn) for rid, (sid, turn) in turns.items() if rid in qa["clue"]
                   and len(turn.get("input_image", [])) == 1
                   and sha256_file(turn["input_image"][0]) == sample.source_sha256]
        if len(targets) != 1:
            raise ValueError("源图未唯一匹配问题证据轮次。")
        target_answer = sample.raw.get("target_answer")
        if not isinstance(target_answer, str) or not target_answer.strip():
            raise ValueError("投毒清单需提供显式 target_answer；目标图片描述不等于目标答案。")
        selected.append({**sample.raw, "family": "poisoning", "qa_id": qa["qa_id"],
                         "session_id": targets[0][0], "turn_id": str(targets[0][1]["round"]),
                         "category": qa["point"], "answer": qa["answer"], "target_answer": target_answer})
    if len({s["turn_id"] for s in selected}) != len(selected):
        raise ValueError("多个目标作用于同一原始轮次，请拆分实验计划以避免相互覆盖。")
    plan = {"family": "poisoning", "dataset": bundle["audit"]["dataset"], "samples": selected,
            "selection_provenance": "existing_frozen_manifest", "manifest_sha256": sha256_file(manifest_path)}
    save_json(output, plan)
    return plan
