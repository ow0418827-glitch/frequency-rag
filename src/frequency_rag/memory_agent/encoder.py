"""用于记忆检索的图文编码，与攻击代理模型分离。"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from frequency_rag.evaluation.vectors import MultimodalVectorEvaluator
from frequency_rag.common.numerics import normalize


class MemoryEncoder:
    def __init__(self, config: dict):
        self.model = MultimodalVectorEvaluator(
            config["model"], device=config.get("device", "cuda"),
            precision=config.get("precision", "fp16"),
            allow_downloads=config.get("allow_downloads", False))

    @torch.no_grad()
    def encode(self, text: str, images=()):
        if not images:
            return self.model.encode_text(text)
        loaded = []
        for path in images:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((224, 224))
                loaded.append(im.copy())
        if self.model.is_gme:
            if text.strip():
                vectors = self.model.model.get_fused_embeddings(texts=[text] * len(loaded), images=loaded)
            else:
                vectors = self.model.model.get_image_embeddings(loaded)
            if torch.is_tensor(vectors):
                vectors = vectors.detach().float().cpu().numpy()
            return normalize(np.asarray(vectors, dtype=np.float32).mean(axis=0))
        # CLIP 消融使用图文向量均值，明确区别于 GME 联合编码。
        vectors = [self.model.encode_image(im) for im in loaded]
        if text.strip():
            vectors.insert(0, self.model.encode_text(text))
        return normalize(np.mean(vectors, axis=0))

    def snapshot(self):
        return {**self.model.snapshot(), "role": "memory_encoder",
                "fusion": "joint_gme" if self.model.is_gme else "normalized_mean_clip_ablation",
                "image_max_side": 224}
