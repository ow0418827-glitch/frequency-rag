from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """以 UTF-8 原子写入结构化数据，避免中途中断留下半个文件。"""
    output = ensure_parent(path)
    handle_id, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(handle_id, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=indent, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_image(path: str | Path | Image.Image) -> Image.Image:
    if isinstance(path, Image.Image):
        return path.convert("RGB").copy()
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def pil_to_tensor(image: Image.Image, device: str | torch.device = "cpu") -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    return tensor.div_(255.0).to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError("图片张量必须具有 [1, 3, H, W] 形状。")
    image = tensor.detach().clamp(0, 1).cpu()[0]
    array = image.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def save_tensor_png(path: str | Path, tensor: torch.Tensor) -> Path:
    output = ensure_parent(path)
    tensor_to_pil(tensor).save(output, format="PNG")
    return output


def save_vector(path: str | Path, vector: np.ndarray) -> Path:
    output = ensure_parent(path)
    np.save(output, np.asarray(vector, dtype=np.float32))
    return output


def decoded_image_audit(
    source: str | Path | Image.Image,
    saved: str | Path | Image.Image,
    *,
    maximum_levels: int = 16,
) -> dict[str, Any]:
    source_array = np.asarray(load_image(source), dtype=np.int16)
    saved_array = np.asarray(load_image(saved), dtype=np.int16)
    if source_array.shape != saved_array.shape:
        raise ValueError(
            f"落盘图片尺寸或通道改变：源图 {source_array.shape}，输出 {saved_array.shape}。"
        )
    difference = saved_array - source_array
    absolute = np.abs(difference)
    maximum = int(absolute.max(initial=0))
    return {
        "shape": list(source_array.shape),
        "max_absolute_pixel_difference_levels": maximum,
        "mean_absolute_pixel_difference_levels": float(absolute.mean()),
        "mean_squared_pixel_difference_levels": float(np.square(difference.astype(np.float64)).mean()),
        "budget_levels": int(maximum_levels),
        "within_budget": maximum <= int(maximum_levels),
    }

