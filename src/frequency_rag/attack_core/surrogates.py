from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from frequency_rag.common.config import SurrogateConfig
from frequency_rag.common.io import sha256_file, stable_key


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class ImageFeatures:
    global_embedding: torch.Tensor
    patch_tokens: torch.Tensor


def resolve_device(requested: str, *, allow_fallback: bool = False) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if allow_fallback:
            return torch.device("cpu")
        raise RuntimeError("配置要求使用图形处理器，但当前张量框架未检测到可用设备。")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            if allow_fallback:
                return torch.device("cpu")
            raise RuntimeError(f"配置要求的图形处理器序号不存在：{device.index}")
    return device


def _huggingface_hub_roots(cache_dir: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir:
        candidate = Path(cache_dir)
        roots.append(candidate if candidate.name == "hub" else candidate / "hub")
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def resolve_hf_cached_weight(
    hf_repo: str | None,
    filename: str = "open_clip_model.safetensors",
    cache_dir: str | Path | None = None,
) -> Path | None:
    if not hf_repo:
        return None
    repository_name = f"models--{hf_repo.replace('/', '--')}"
    matches: list[Path] = []
    for hub_root in _huggingface_hub_roots(cache_dir):
        repository = hub_root / repository_name
        if not repository.exists():
            continue
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            candidate = repository / "snapshots" / revision / filename
            if candidate.is_file():
                return candidate
        snapshots = repository / "snapshots"
        if snapshots.exists():
            matches.extend(path for path in snapshots.glob(f"*/{filename}") if path.is_file())
    return sorted(matches, key=str)[-1] if matches else None


def resolve_hf_snapshot(hf_repo: str, cache_dir: str | Path | None = None) -> Path | None:
    repository_name = f"models--{hf_repo.replace('/', '--')}"
    candidates: list[Path] = []
    for hub_root in _huggingface_hub_roots(cache_dir):
        repository = hub_root / repository_name
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            snapshot = repository / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot.resolve()
        snapshots = repository / "snapshots"
        if snapshots.is_dir():
            candidates.extend(path.resolve() for path in snapshots.iterdir() if path.is_dir())
    return sorted(candidates, key=str)[-1] if candidates else None


def _infer_image_size(model: torch.nn.Module) -> int:
    image_size = getattr(getattr(model, "visual", None), "image_size", 224)
    if isinstance(image_size, Iterable) and not isinstance(image_size, (str, bytes)):
        return int(tuple(image_size)[0])
    return int(image_size)


def _revision_from_weight_path(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None

def _is_safetensors_file(path: str | Path | None) -> bool:
    if not path:
        return False
    path_obj = Path(path)
    if path_obj.name.endswith(".safetensors") or str(path).endswith(".safetensors"):
        return True
    try:
        if path_obj.is_file():
            with open(path_obj, "rb") as f:
                header = f.read(9)
            if len(header) >= 9:
                import struct
                header_size = struct.unpack("<Q", header[:8])[0]
                if 0 < header_size < 100 * 1024 * 1024 and header[8:9] == b"{":
                    return True
    except Exception:
        pass
    return False


def _patch_open_clip_safetensors() -> None:
    try:
        import open_clip.factory
    except ImportError:
        return

    orig_load_state_dict = getattr(open_clip.factory, "load_state_dict", None)
    if orig_load_state_dict is None or getattr(orig_load_state_dict, "_safetensors_patched", False):
        return

    def safe_load_state_dict(checkpoint_path: str, device="cpu", weights_only=True):
        if _is_safetensors_file(checkpoint_path):
            from safetensors.torch import load_file
            checkpoint = load_file(str(checkpoint_path), device=str(device))
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            if next(iter(state_dict.items()))[0].startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            return state_dict

        try:
            return orig_load_state_dict(checkpoint_path, device=device, weights_only=weights_only)
        except TypeError:
            return orig_load_state_dict(checkpoint_path, device=device)

    safe_load_state_dict._safetensors_patched = True
    open_clip.factory.load_state_dict = safe_load_state_dict


class OpenCLIPSurrogate:
    """参考项目代理模型的独立兼容包装，并显式拒绝局部特征静默回退。"""

    def __init__(
        self,
        config: SurrogateConfig,
        *,
        device: str = "cuda",
        precision: str = "fp32",
        allow_downloads: bool = False,
        allow_device_fallback: bool = False,
        require_true_local_tokens: bool = True,
        hash_weight: bool = True,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                f"无法导入 open_clip_torch 或其运行依赖：{exc}"
            ) from exc
        if precision != "fp32":
            raise ValueError("首轮攻击代理只允许 fp32 精度。")
        self.config = config
        self.device = resolve_device(device, allow_fallback=allow_device_fallback)
        self.precision = precision
        self.require_true_local_tokens = bool(require_true_local_tokens)

        weight_path: Path | None = None
        if config.weight_path:
            weight_path = Path(config.weight_path)
            if not weight_path.is_file():
                raise FileNotFoundError(f"显式指定的代理权重不存在：{weight_path}")
        else:
            weight_path = resolve_hf_cached_weight(config.hf_repo)
        if weight_path is None and not allow_downloads:
            raise FileNotFoundError(
                f"没有找到 {config.name}/{config.pretrained} 的本地冻结权重；"
                "当前配置禁止自动下载。请显式提供 weight_path 或允许下载。"
            )
        _patch_open_clip_safetensors()
        pretrained = str(weight_path) if weight_path else config.pretrained
        try:
            self.model = open_clip.create_model(config.name, pretrained=pretrained)
        except Exception as exc:
            if _is_safetensors_file(weight_path):
                try:
                    from safetensors.torch import load_file
                    self.model = open_clip.create_model(config.name, pretrained=None)
                    state_dict = load_file(str(weight_path), device="cpu")
                    if next(iter(state_dict.items()))[0].startswith("module."):
                        state_dict = {k[7:]: v for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict, strict=False)
                except Exception:
                    raise exc
            elif "weights_only" in str(exc) or "UnpicklingError" in type(exc).__name__:
                try:
                    self.model = open_clip.create_model(
                        config.name, pretrained=pretrained, weights_only=False
                    )
                except TypeError:
                    raise exc
            else:
                raise
        self.model.eval().to(self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(config.name)
        self.image_size = _infer_image_size(self.model)
        self.mean = torch.tensor(
            CLIP_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            CLIP_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.weight_path = weight_path
        self.weight_sha256 = sha256_file(weight_path) if weight_path and hash_weight else None
        self.weight_revision = _revision_from_weight_path(weight_path)
        self._open_clip_version = importlib.metadata.version("open-clip-torch")
        self._cache_identity = stable_key(
            (
                config.name,
                config.pretrained,
                config.hf_repo or "",
                str(weight_path or "downloaded-by-open-clip"),
                self.weight_sha256 or "hash-not-recorded",
                self.weight_revision or "revision-unavailable",
                precision,
                str(self.image_size),
                json.dumps(CLIP_MEAN),
                json.dumps(CLIP_STD),
                self._open_clip_version,
            )
        )

    @property
    def cache_identity(self) -> str:
        return self._cache_identity

    def snapshot(self) -> dict[str, Any]:
        return {
            "role": "attack_surrogate",
            "name": self.config.name,
            "pretrained": self.config.pretrained,
            "hf_repo": self.config.hf_repo,
            "weight_path": str(self.weight_path) if self.weight_path else None,
            "weight_sha256": self.weight_sha256,
            "weight_revision": self.weight_revision,
            "device": str(self.device),
            "precision": self.precision,
            "image_size": self.image_size,
            "preprocess": {
                "resize": "bicubic_direct_square",
                "align_corners": False,
                "antialias": True,
                "mean": list(CLIP_MEAN),
                "std": list(CLIP_STD),
            },
            "open_clip_torch_version": self._open_clip_version,
            "cache_identity": self.cache_identity,
            "requires_true_local_tokens": self.require_true_local_tokens,
        }

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("代理模型输入必须具有 [批, 3, 高, 宽] 形状。")
        image = images.to(self.device, dtype=torch.float32)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (image - self.mean) / self.std

    def encode_image(self, images: torch.Tensor) -> ImageFeatures:
        normalized = self.preprocess(images)
        visual = self.model.visual
        if not hasattr(visual, "forward_intermediates"):
            if self.require_true_local_tokens:
                raise RuntimeError(
                    f"代理 {self.config.name}/{self.config.pretrained} 没有局部特征接口，兼容实验已中止。"
                )
            global_embedding = self.model.encode_image(normalized, normalize=True)
            return ImageFeatures(global_embedding, global_embedding[:, None, :])

        output = visual.forward_intermediates(
            normalized,
            indices=[-1],
            output_fmt="NLC",
            output_extra_tokens=True,
        )
        if not isinstance(output, dict):
            raise RuntimeError("当前 open_clip_torch 返回了未知的局部特征结构。")
        try:
            global_embedding = F.normalize(output["image_features"], dim=-1)
            patch_tokens = F.normalize(output["image_intermediates"][-1], dim=-1)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("代理模型局部特征接口缺少预期字段。") from exc
        if patch_tokens.ndim != 3:
            raise RuntimeError(f"代理模型局部特征形状异常：{tuple(patch_tokens.shape)}")
        if self.require_true_local_tokens and patch_tokens.shape[1] <= 1:
            raise RuntimeError("代理模型只返回了一个全局向量，不能冒充局部图块特征。")
        return ImageFeatures(global_embedding, patch_tokens)

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        texts = [text] if isinstance(text, str) else text
        tokens = self.tokenizer(texts).to(self.device)
        return F.normalize(self.model.encode_text(tokens), dim=-1)


def load_surrogates(
    configs: Iterable[SurrogateConfig],
    *,
    device: str = "cuda",
    precision: str = "fp32",
    allow_downloads: bool = False,
    allow_device_fallback: bool = False,
    require_true_local_tokens: bool = True,
    hash_weights: bool = True,
) -> list[OpenCLIPSurrogate]:
    return [
        OpenCLIPSurrogate(
            config,
            device=device,
            precision=precision,
            allow_downloads=allow_downloads,
            allow_device_fallback=allow_device_fallback,
            require_true_local_tokens=require_true_local_tokens,
            hash_weight=hash_weights,
        )
        for config in configs
    ]
