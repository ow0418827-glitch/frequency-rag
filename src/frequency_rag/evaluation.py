from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .io import load_image, save_json, save_vector, sha256_file
from .models import resolve_device, resolve_hf_snapshot
from .profiling import DeviceMemoryMonitor, PhaseClock


ENCODER_MODEL_IDS = {
    "CLIP-B-32": "openai/clip-vit-base-patch32",
    "CLIP": "openai/clip-vit-base-patch32",
    "CLIP-BASE": "openai/clip-vit-base-patch32",
    "CLIP-336": "openai/clip-vit-large-patch14-336",
    "CLIP-LARGE": "openai/clip-vit-large-patch14-336",
    "SIGLIP": "google/siglip-base-patch16-224",
}


def normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norm, 1e-12, None)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalize(left), normalize(right)))


def l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(normalize(left) - normalize(right)))


def compute_vector_metrics(
    *,
    sample_id: str,
    encoder_name: str,
    encoder_model_id: str,
    query_text: str,
    source_vector: np.ndarray,
    adversarial_vector: np.ndarray,
    target_vector: np.ndarray,
    query_vector: np.ndarray,
) -> dict[str, Any]:
    source = normalize(np.asarray(source_vector, dtype=np.float32))
    adversarial = normalize(np.asarray(adversarial_vector, dtype=np.float32))
    target = normalize(np.asarray(target_vector, dtype=np.float32))
    query = normalize(np.asarray(query_vector, dtype=np.float32))
    source_target = cosine_similarity(source, target)
    adversarial_target = cosine_similarity(adversarial, target)
    source_query = cosine_similarity(source, query)
    adversarial_query = cosine_similarity(adversarial, query)
    source_target_distance = l2_distance(source, target)
    adversarial_target_distance = l2_distance(adversarial, target)
    return {
        "sample_id": sample_id,
        "encoder": encoder_name,
        "encoder_model_id": encoder_model_id,
        "embedding_dimension": int(source.size),
        "query_text": query_text,
        "cosine_metrics": {
            "source_target_cosine": source_target,
            "adversarial_target_cosine": adversarial_target,
            "target_cosine_gain": adversarial_target - source_target,
            "source_query_cosine": source_query,
            "adversarial_query_cosine": adversarial_query,
            "query_cosine_gain": adversarial_query - source_query,
            "source_adversarial_cosine": cosine_similarity(source, adversarial),
        },
        "distance_metrics": {
            "source_target_l2_distance": source_target_distance,
            "adversarial_target_l2_distance": adversarial_target_distance,
            "target_l2_distance_reduction": source_target_distance - adversarial_target_distance,
        },
    }


class MultimodalVectorEvaluator:
    """只在离线评估阶段加载，攻击循环无法访问此对象。"""

    def __init__(
        self,
        encoder_name: str = "CLIP-B-32",
        *,
        expected_model_id: str | None = None,
        device: str = "cuda",
        precision: str = "fp16",
        allow_downloads: bool = False,
        allow_device_fallback: bool = False,
        hash_weights: bool = True,
    ) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("离线向量评估需要安装 transformers。") from exc
        self.encoder_name = encoder_name
        self.device = resolve_device(device, allow_fallback=allow_device_fallback)
        normalized_name = encoder_name.upper()
        self.model_id = ENCODER_MODEL_IDS.get(normalized_name, encoder_name)
        if expected_model_id and self.model_id != expected_model_id:
            raise ValueError(
                f"评估模型标识不一致：解析为 {self.model_id}，配置要求 {expected_model_id}。"
            )
        self.is_gme = "gme" in self.model_id.lower()
        if precision == "fp16" and self.device.type == "cuda":
            dtype = torch.float16
            effective_precision = "fp16"
        else:
            dtype = torch.float32
            effective_precision = "fp32"
        self.precision = effective_precision
        local_only = not allow_downloads
        if self.is_gme:
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                local_files_only=local_only,
            ).eval()
            self.processor = None
        else:
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                local_files_only=local_only,
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                local_files_only=local_only,
                use_fast=False,
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.snapshot_directory = resolve_hf_snapshot(self.model_id)
        self.weight_files: list[dict[str, Any]] = []
        if self.snapshot_directory and hash_weights:
            patterns = ("*.safetensors", "*.bin")
            paths: list[Path] = []
            for pattern in patterns:
                paths.extend(self.snapshot_directory.glob(pattern))
            for path in sorted(set(paths), key=str):
                if path.is_file():
                    self.weight_files.append(
                        {
                            "path": str(path.resolve()),
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                    )

    def snapshot(self) -> dict[str, Any]:
        revision = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        return {
            "role": "offline_primary_evaluator",
            "encoder_name": self.encoder_name,
            "model_id": self.model_id,
            "revision": revision,
            "snapshot_directory": str(self.snapshot_directory) if self.snapshot_directory else None,
            "weight_files": self.weight_files,
            "device": str(self.device),
            "precision": self.precision,
            "transformers_version": importlib.metadata.version("transformers"),
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
            "processor_use_fast": False if self.processor is not None else None,
            "model_class": type(self.model).__name__,
        }

    @torch.no_grad()
    def encode_image(self, image: str | Path | Image.Image) -> np.ndarray:
        loaded = load_image(image)
        if self.is_gme:
            embedding = self.model.get_image_embeddings([loaded])
            if torch.is_tensor(embedding):
                embedding = embedding.detach().float().cpu().numpy()
            return normalize(np.asarray(embedding[0], dtype=np.float32))
        inputs = self.processor(images=loaded, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        features = self.model.get_image_features(**inputs)
        if not torch.is_tensor(features):
            features = features.image_embeds if hasattr(features, "image_embeds") else features[0]
        return normalize(np.asarray(features[0].detach().float().cpu(), dtype=np.float32))

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        normalized_text = str(text or " ").strip() or " "
        if self.is_gme:
            embedding = self.model.get_text_embeddings([normalized_text])
            if torch.is_tensor(embedding):
                embedding = embedding.detach().float().cpu().numpy()
            return normalize(np.asarray(embedding[0], dtype=np.float32))
        inputs = self.processor(
            text=[normalized_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        features = self.model.get_text_features(**inputs)
        if not torch.is_tensor(features):
            features = features.text_embeds if hasattr(features, "text_embeds") else features[0]
        return normalize(np.asarray(features[0].detach().float().cpu(), dtype=np.float32))

    def evaluate_sample(
        self,
        *,
        sample_id: str,
        query_text: str,
        source_image: str | Path | Image.Image,
        adversarial_image: str | Path | Image.Image,
        target_image: str | Path | Image.Image,
        output_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        source_vector = self.encode_image(source_image)
        adversarial_vector = self.encode_image(adversarial_image)
        target_vector = self.encode_image(target_image)
        query_vector = self.encode_text(query_text)
        result = compute_vector_metrics(
            sample_id=sample_id,
            encoder_name=self.encoder_name,
            encoder_model_id=self.model_id,
            query_text=query_text,
            source_vector=source_vector,
            adversarial_vector=adversarial_vector,
            target_vector=target_vector,
            query_vector=query_vector,
        )
        if output_directory is not None:
            output = Path(output_directory)
            save_vector(output / "source_vector.npy", source_vector)
            save_vector(output / "adversarial_vector.npy", adversarial_vector)
            save_vector(output / "target_vector.npy", target_vector)
            save_vector(output / "query_vector.npy", query_vector)
            save_json(output / "vector_evaluation.json", result)
        return result


def measured_evaluation(
    evaluator: MultimodalVectorEvaluator,
    *,
    sample_id: str,
    query_text: str,
    source_image: str | Path,
    adversarial_image: str | Path,
    target_image: str | Path,
    output_directory: str | Path,
    memory_sample_interval_seconds: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    monitor = DeviceMemoryMonitor(evaluator.device, memory_sample_interval_seconds)
    monitor.start()
    clock = PhaseClock(evaluator.device)
    try:
        clock.start()
        result = evaluator.evaluate_sample(
            sample_id=sample_id,
            query_text=query_text,
            source_image=source_image,
            adversarial_image=adversarial_image,
            target_image=target_image,
            output_directory=output_directory,
        )
        elapsed = clock.stop()
        memory = monitor.stop()
    finally:
        monitor.cancel()
    return result, {"evaluation_seconds": elapsed, "memory": memory}
