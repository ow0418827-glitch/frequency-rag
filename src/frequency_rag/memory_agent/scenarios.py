"""核验落盘扰动图，构造隔离的实验条件。"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from PIL import Image
from frequency_rag.common.io import load_json, sha256_file


def adversarial_images(run_dir, samples, epsilon=16 / 255):
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("无效的像素扰动上限。")
    root = Path(run_dir).resolve()
    manifest = load_json(root / "run_manifest.json")
    rows = {r["sample_id"]: r for r in manifest["samples"]}
    if len(rows) != len(manifest["samples"]):
        raise ValueError("扰动运行包含重复样本标识。")
    result = {}
    for sample in samples:
        row = rows[sample["sample_id"]]
        if row["status"] not in {"generated", "evaluated"}:
            raise ValueError("扰动运行未成功完成。")
        path = Path(row["adversarial_image"])
        if not path.is_absolute():
            path = root / path
        expected = row.get("adversarial_sha256")
        if not expected or sha256_file(path) != expected:
            raise ValueError("对抗图片摘要缺失或内容已经改变。")
        for key in ("source", "target"):
            if sha256_file(sample[key + "_image"]) != sample[key + "_sha256"]:
                raise ValueError("实验计划中的图片发生变化。")
            if row.get(key + "_sha256") != sample[key + "_sha256"]:
                raise ValueError("扰动运行与实验计划的源／目标图不匹配。")
        with Image.open(sample["source_image"]) as im:
            source = np.array(im.convert("RGB"), dtype=np.int16)
        with Image.open(path) as im:
            adv = np.array(im.convert("RGB"), dtype=np.int16)
        if source.shape != adv.shape or np.abs(source - adv).max() > np.ceil(epsilon * 255):
            raise ValueError("对抗图片尺寸或像素扰动上限不符合计划。")
        result[sample["sample_id"]] = str(path.resolve())
    return result


def condition_dialog(bundle, plan, condition, *, images=None, describer=None):
    if condition not in {"clean", "oracle", "adversarial", "source_injection_control"}:
        raise ValueError("未知实验条件。")
    dialog = copy.deepcopy(bundle["dialog"])
    if condition == "clean":
        return dialog
    for item in plan["samples"]:
        if condition == "adversarial":
            path = images[item["sample_id"]]
        else:
            path = item["target_image" if condition == "oracle" else "source_image"]
        if plan["family"] == "injection":
            matches = [s for s in dialog["multi_session_dialogues"] if str(s["session_id"]) == item["session_id"]]
            if len(matches) != 1 or matches[0] is dialog["multi_session_dialogues"][-1]:
                raise ValueError("注入会话不存在、重复或位于最后会话。")
            turn = {"round": item["turn_id"], "user": item["user"], "assistant": "",
                    "image_id": [item["turn_id"] + ":IMG"]}
            matches[0]["dialogues"].append(turn)
        elif plan["family"] == "poisoning":
            matches = [t for s in dialog["multi_session_dialogues"] for t in s["dialogues"]
                       if str(t["round"]) == item["turn_id"]]
            if len(matches) != 1:
                raise ValueError("投毒轮次不存在或重复。")
            turn = matches[0]
            if len(turn.get("input_image", [])) != 1:
                raise ValueError("投毒计划当前要求源轮次恰有一张图，禁止误删其他图像。")
            if sha256_file(turn["input_image"][0]) != item["source_sha256"]:
                raise ValueError("投毒计划源图与原会话不一致。")
        else:
            raise ValueError("未知攻击类型。")
        turn["input_image"] = [path]
        # 对抗条件仅使用受害模型从实际图像生成的描述，不复制目标文本。
        caption = item["target_text"] if condition == "oracle" else describer.describe([path])
        turn["image_caption"] = [caption]
    return dialog
