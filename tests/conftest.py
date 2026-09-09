from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from frequency_rag.config import ProjectConfig, load_config
from frequency_rag.models import ImageFeatures


class TinySurrogate:
    def __init__(self, index: int) -> None:
        self.index = index
        self.device = torch.device("cpu")
        self.cache_identity = f"tiny-surrogate-{index}"
        self.image_calls = 0
        self.text_calls = 0

    def encode_image(self, image: torch.Tensor) -> ImageFeatures:
        self.image_calls += 1
        means = image.mean(dim=(-2, -1))
        squares = image.square().mean(dim=(-2, -1))
        global_embedding = torch.cat((means, squares), dim=-1)
        global_embedding = torch.roll(global_embedding, self.index, dims=-1)
        patches = F.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
        patch_tokens = torch.cat((patches, patches.square()), dim=-1)
        patch_tokens = torch.roll(patch_tokens, self.index, dims=-1)
        return ImageFeatures(
            F.normalize(global_embedding, dim=-1),
            F.normalize(patch_tokens, dim=-1),
        )

    def encode_text(self, text: str) -> torch.Tensor:
        self.text_calls += 1
        value = str(text)
        numbers = torch.tensor(
            [
                len(value) + 1,
                sum(map(ord, value)) % 97 + 1,
                value.count("a") + 1,
                value.count("e") + 1,
                value.count("i") + 1,
                value.count(" ") + 1,
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        return F.normalize(torch.roll(numbers, self.index, dims=-1), dim=-1)


@pytest.fixture
def tiny_surrogates() -> list[TinySurrogate]:
    return [TinySurrogate(index) for index in range(3)]


@pytest.fixture
def cpu_config() -> ProjectConfig:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.json")
    attack = replace(
        config.attack,
        reference_steps=3,
        clusters=3,
        cluster_iterations=3,
        sinkhorn_iterations=8,
        sinkhorn_tolerance=0.0,
        log_every=1,
    )
    runtime = replace(
        config.runtime,
        allow_downloads=False,
        allow_device_fallback=False,
        hash_model_weights=False,
        device_memory_sample_interval_seconds=0.01,
    )
    return replace(config, device="cpu", attack=attack, runtime=runtime)

