from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from .config import AttackConfig
from .models import ImageFeatures
from .ot import local_ot_similarity, token_kmeans


@dataclass(frozen=True)
class TargetBundle:
    image_features: tuple[ImageFeatures, ...]
    cluster_centers: tuple[torch.Tensor, ...]
    text_feature: torch.Tensor | None
    image_key: str
    text_key: str | None


@dataclass(frozen=True)
class ObjectiveTensors:
    loss: torch.Tensor
    global_similarities: torch.Tensor
    local_similarities: torch.Tensor
    text_similarity: torch.Tensor

    def detached_record(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss.detach().cpu()),
            "global_similarity": [
                float(value) for value in self.global_similarities.detach().cpu().tolist()
            ],
            "local_similarity": [
                float(value) for value in self.local_similarities.detach().cpu().tolist()
            ],
            "text_similarity": float(self.text_similarity.detach().cpu()),
        }


class TargetCache:
    """同一进程内复用确定性的目标特征、目标聚类中心和文本向量。"""

    def __init__(self) -> None:
        self.image_features: dict[str, ImageFeatures] = {}
        self.cluster_centers: dict[str, torch.Tensor] = {}
        self.text_features: dict[str, torch.Tensor] = {}
        self.hits = {"image": 0, "centers": 0, "text": 0}
        self.misses = {"image": 0, "centers": 0, "text": 0}

    def get_image(self, key: str, factory: Callable[[], ImageFeatures]) -> ImageFeatures:
        if key in self.image_features:
            self.hits["image"] += 1
            return self.image_features[key]
        self.misses["image"] += 1
        value = factory()
        detached = ImageFeatures(
            global_embedding=value.global_embedding.detach(),
            patch_tokens=value.patch_tokens.detach(),
        )
        self.image_features[key] = detached
        return detached

    def get_centers(self, key: str, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key in self.cluster_centers:
            self.hits["centers"] += 1
            return self.cluster_centers[key]
        self.misses["centers"] += 1
        value = factory().detach()
        self.cluster_centers[key] = value
        return value

    def get_text(self, key: str, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key in self.text_features:
            self.hits["text"] += 1
            return self.text_features[key]
        self.misses["text"] += 1
        value = factory().detach()
        self.text_features[key] = value
        return value

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            "hits": dict(self.hits),
            "misses": dict(self.misses),
            "entries": {
                "image": len(self.image_features),
                "centers": len(self.cluster_centers),
                "text": len(self.text_features),
            },
        }


def _tensor_key(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()
    return f"sha256:{digest}:shape={tuple(contiguous.shape)}:dtype={contiguous.dtype}"


def _surrogate_identity(surrogate: Any, index: int) -> str:
    identity = getattr(surrogate, "cache_identity", None)
    if callable(identity):
        identity = identity()
    if identity:
        return str(identity)
    config = getattr(surrogate, "config", None)
    return f"{type(surrogate).__module__}.{type(surrogate).__qualname__}:{config!r}:index={index}"


def prepare_targets(
    surrogates: Sequence[Any],
    target_image: torch.Tensor,
    target_text: str | None,
    attack_config: AttackConfig,
    *,
    cache: TargetCache | None = None,
    target_image_key: str | None = None,
    prepare_cluster_centers: bool = True,
) -> TargetBundle:
    if not surrogates:
        raise ValueError("至少需要一个代理模型。")
    shared_cache = cache or TargetCache()
    image_key = target_image_key or _tensor_key(target_image)
    features: list[ImageFeatures] = []
    centers: list[torch.Tensor] = []
    with torch.no_grad():
        for index, surrogate in enumerate(surrogates):
            model_key = _surrogate_identity(surrogate, index)
            feature_key = f"image|{model_key}|{image_key}"
            feature = shared_cache.get_image(
                feature_key, lambda surrogate=surrogate: surrogate.encode_image(target_image)
            )
            features.append(feature)
            if prepare_cluster_centers:
                center_key = (
                    f"centers|{feature_key}|clusters={attack_config.clusters}"
                    f"|iterations={attack_config.cluster_iterations}"
                )
                center = shared_cache.get_centers(
                    center_key,
                    lambda feature=feature: token_kmeans(
                        feature.patch_tokens,
                        attack_config.clusters,
                        attack_config.cluster_iterations,
                    ),
                )
                centers.append(center)

        text_feature: torch.Tensor | None = None
        text_key: str | None = None
        if target_text:
            text_index = attack_config.text_model_index
            model_key = _surrogate_identity(surrogates[text_index], text_index)
            text_key = f"text|{model_key}|{target_text}"
            text_feature = shared_cache.get_text(
                text_key,
                lambda: surrogates[text_index].encode_text(target_text),
            )
    return TargetBundle(
        image_features=tuple(features),
        cluster_centers=tuple(centers),
        text_feature=text_feature,
        image_key=image_key,
        text_key=text_key,
    )


def _one_model_terms(
    feature: ImageFeatures,
    target_feature: ImageFeatures,
    target_center: torch.Tensor | None,
    attack_config: AttackConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_similarity = F.cosine_similarity(
        feature.global_embedding, target_feature.global_embedding
    ).mean()
    local_similarity = local_ot_similarity(
        feature.patch_tokens,
        target_tokens=target_feature.patch_tokens if target_center is None else None,
        target_centers=target_center,
        clusters=attack_config.clusters,
        cluster_iterations=attack_config.cluster_iterations,
        sinkhorn_iterations=attack_config.sinkhorn_iterations,
        sinkhorn_regularization=attack_config.sinkhorn_regularization,
        sinkhorn_tolerance=attack_config.sinkhorn_tolerance,
        detach_transport_plan=attack_config.detach_transport_plan,
    ).mean()
    return global_similarity, local_similarity


def joint_objective(
    image: torch.Tensor,
    surrogates: Sequence[Any],
    targets: TargetBundle,
    model_weights: torch.Tensor,
    attack_config: AttackConfig,
    *,
    use_cached_target_centers: bool = True,
) -> ObjectiveTensors:
    if len(surrogates) != len(targets.image_features):
        raise ValueError("代理模型与目标特征数量不一致。")
    if use_cached_target_centers and len(targets.cluster_centers) != len(surrogates):
        raise ValueError("启用目标中心缓存时，中心数量必须与代理模型一致。")
    features = [surrogate.encode_image(image) for surrogate in surrogates]
    terms = [
        _one_model_terms(
            feature,
            target,
            targets.cluster_centers[index] if use_cached_target_centers else None,
            attack_config,
        )
        for index, (feature, target) in enumerate(
            zip(features, targets.image_features, strict=True)
        )
    ]
    global_similarities = torch.stack([term[0] for term in terms])
    local_similarities = torch.stack([term[1] for term in terms])
    text_similarity = image.new_zeros(())
    if targets.text_feature is not None and attack_config.text_weight:
        text_similarity = F.cosine_similarity(
            features[attack_config.text_model_index].global_embedding,
            targets.text_feature,
        ).mean()
    loss = (model_weights * global_similarities).sum()
    loss = loss + float(attack_config.local_weight) * local_similarities.mean()
    loss = loss + float(attack_config.text_weight) * text_similarity
    return ObjectiveTensors(loss, global_similarities, local_similarities, text_similarity)


def sequential_image_gradient(
    image_leaf: torch.Tensor,
    surrogates: Sequence[Any],
    targets: TargetBundle,
    model_weights: torch.Tensor,
    attack_config: AttackConfig,
) -> tuple[torch.Tensor, ObjectiveTensors]:
    """逐代理释放主干计算图，仅累计同一图片状态的空间梯度。"""
    if not image_leaf.is_leaf or not image_leaf.requires_grad:
        raise ValueError("逐代理梯度入口必须是需要梯度的叶子图片张量。")
    count = len(surrogates)
    if count == 0 or count != len(targets.image_features) or len(targets.cluster_centers) not in {0, count}:
        raise ValueError("代理模型与目标特征数量不一致。")
    accumulated = torch.zeros_like(image_leaf)
    global_values: list[torch.Tensor] = []
    local_values: list[torch.Tensor] = []
    text_value = image_leaf.new_zeros(())
    loss_value = image_leaf.new_zeros(())

    for index, surrogate in enumerate(surrogates):
        feature = surrogate.encode_image(image_leaf)
        global_similarity, local_similarity = _one_model_terms(
            feature,
            targets.image_features[index],
            targets.cluster_centers[index] if targets.cluster_centers else None,
            attack_config,
        )
        contribution = model_weights[index] * global_similarity
        contribution = contribution + float(attack_config.local_weight) * local_similarity / count
        current_text = image_leaf.new_zeros(())
        if (
            index == attack_config.text_model_index
            and targets.text_feature is not None
            and attack_config.text_weight
        ):
            current_text = F.cosine_similarity(
                feature.global_embedding, targets.text_feature
            ).mean()
            contribution = contribution + float(attack_config.text_weight) * current_text
            text_value = current_text.detach()
        gradient = torch.autograd.grad(contribution, image_leaf, retain_graph=False)[0]
        accumulated.add_(gradient.detach())
        loss_value = loss_value + contribution.detach()
        global_values.append(global_similarity.detach())
        local_values.append(local_similarity.detach())
        del feature, contribution, gradient

    metrics = ObjectiveTensors(
        loss=loss_value,
        global_similarities=torch.stack(global_values),
        local_similarities=torch.stack(local_values),
        text_similarity=text_value,
    )
    return accumulated, metrics


@torch.no_grad()
def evaluate_objective(
    image: torch.Tensor,
    surrogates: Sequence[Any],
    targets: TargetBundle,
    model_weights: torch.Tensor,
    attack_config: AttackConfig,
    *,
    use_cached_target_centers: bool = True,
) -> ObjectiveTensors:
    return joint_objective(
        image,
        surrogates,
        targets,
        model_weights,
        attack_config,
        use_cached_target_centers=use_cached_target_centers,
    )
