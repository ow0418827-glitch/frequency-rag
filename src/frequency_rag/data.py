from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import load_json, sha256_file


MEMGALLERY_DATASETS = (
    "AI_Robotics_Automation_Future_Tech",
    "Technology_Ethics_Future_Society",
    "Parenting_Commuting_Hobbies_Travel_Gear",
    "Home_Repair_Maintenance_Cleaning",
    "Real_Estate_Home_Decor_DIY_Lifestyle",
)


@dataclass(frozen=True)
class MemGalleryCandidate:
    candidate_id: str
    dataset: str
    session_id: str
    round_id: str
    image_path: Path
    caption: str


@dataclass(frozen=True)
class MemGalleryQuery:
    query_id: str
    dataset: str
    session_id: str
    question: str
    answer: str
    source_image_path: Path
    category: str
    source_caption: str


def _resolve_image_candidate(
    raw_path: str | Path,
    base_dir: Path | None = None,
    sample_id: str | None = None,
    role: str | None = None,
) -> Path:
    p = Path(raw_path)
    if p.is_absolute() and p.is_file():
        return p.resolve()

    project_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []

    if not p.is_absolute():
        if base_dir:
            candidates.append((base_dir / p).resolve())
        candidates.append((project_root / p).resolve())
        candidates.append(p.resolve())

    # Check data/ by sample_id and role (source.jpg or target.jpg)
    if sample_id and role:
        candidates.append((project_root / "data" / sample_id / f"{role}.jpg").resolve())
        candidates.append((project_root / "data" / sample_id / f"{role}.png").resolve())
        if base_dir:
            candidates.append((base_dir / sample_id / f"{role}.jpg").resolve())
            candidates.append((base_dir / sample_id / f"{role}.png").resolve())

    # Check data/ by filename
    candidates.append((project_root / "data" / p.name).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if base_dir and not p.is_absolute():
        return (base_dir / p).resolve()
    return p.resolve()


@dataclass(frozen=True)
class FrozenSample:
    sample_id: str
    question: str
    answer: str
    source_image: Path
    source_sha256: str
    target_image: Path
    target_sha256: str
    target_text: str
    selection_score: float
    phi_r: float
    phi_contra: float
    phi_c: float
    raw: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        base_dir: Path | None = None,
    ) -> "FrozenSample":
        sid = str(raw["sample_id"])
        src = _resolve_image_candidate(raw["source_image"], base_dir=base_dir, sample_id=sid, role="source")
        tgt = _resolve_image_candidate(raw["target_image"], base_dir=base_dir, sample_id=sid, role="target")
        return cls(
            sample_id=sid,
            question=str(raw["question"]),
            answer=str(raw.get("answer", "")),
            source_image=src,
            source_sha256=str(raw["source_sha256"]),
            target_image=tgt,
            target_sha256=str(raw["target_sha256"]),
            target_text=str(raw.get("target_text", "")),
            selection_score=float(raw["selection_score"]),
            phi_r=float(raw["phi_r"]),
            phi_contra=float(raw["phi_contra"]),
            phi_c=float(raw["phi_c"]),
            raw=dict(raw),
        )


def _clean_session_id(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "_".join(str(item) for item in value)
    return str(value) if value is not None else ""


def resolve_image_path(memgallery_root: Path, dataset: str, raw_image: str) -> Path | None:
    path = Path(raw_image)
    if path.is_absolute() and path.exists():
        return path.resolve()
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("../image/"):
        normalized = normalized[len("../image/") :]
    candidates = (
        memgallery_root / "benchmark" / "data" / "image" / dataset / Path(normalized).name,
        memgallery_root / "benchmark" / "data" / "image" / normalized,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_candidates_from_memgallery(
    memgallery_root: str | Path,
    datasets: Iterable[str] | None = None,
) -> list[MemGalleryCandidate]:
    root = Path(memgallery_root).resolve()
    target_datasets = tuple(datasets or MEMGALLERY_DATASETS)
    candidates: list[MemGalleryCandidate] = []
    seen_paths: set[str] = set()
    for dataset in target_datasets:
        dialog_path = root / "benchmark" / "data" / "dialog" / f"{dataset}.json"
        if not dialog_path.exists():
            continue
        data = load_json(dialog_path)
        for session in data.get("multi_session_dialogues", []):
            session_id = _clean_session_id(session.get("session_id", ""))
            for turn in session.get("dialogues", []):
                round_id = str(turn.get("round", ""))
                images = turn.get("input_image", []) or []
                if isinstance(images, str):
                    images = [images]
                user_text = str(turn.get("user", "")).strip()
                assistant_text = str(turn.get("assistant", "")).strip()
                caption = f"{user_text} {assistant_text}".strip() or "A photo shared in the conversation."
                for index, raw_image in enumerate(images):
                    resolved = resolve_image_path(root, dataset, str(raw_image))
                    if resolved is None:
                        continue
                    key = str(resolved)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    candidates.append(
                        MemGalleryCandidate(
                            candidate_id=f"{dataset}:{session_id}:{round_id}:{index}",
                            dataset=dataset,
                            session_id=session_id,
                            round_id=round_id,
                            image_path=resolved,
                            caption=caption,
                        )
                    )
    return candidates


def load_queries_from_memgallery(memgallery_root: str | Path, dataset: str) -> list[MemGalleryQuery]:
    root = Path(memgallery_root).resolve()
    dialog_path = root / "benchmark" / "data" / "dialog" / f"{dataset}.json"
    if not dialog_path.exists():
        raise FileNotFoundError(f"对话数据文件不存在：{dialog_path}")
    data = load_json(dialog_path)

    # 与冻结参考实现一致：只用轮次作键，后遇到的同键记录覆盖前者。
    round_lookup: dict[str, tuple[Path, str]] = {}
    for session in data.get("multi_session_dialogues", []):
        for turn in session.get("dialogues", []):
            round_id = str(turn.get("round", ""))
            images = turn.get("input_image", []) or []
            if isinstance(images, str):
                images = [images]
            caption = f"{turn.get('user', '')} {turn.get('assistant', '')}".strip()
            for raw_image in images:
                resolved = resolve_image_path(root, dataset, str(raw_image))
                if resolved is not None:
                    round_lookup[round_id] = (resolved, caption)
                    break

    queries: list[MemGalleryQuery] = []
    for index, qa in enumerate(data.get("human-annotated QAs", [])):
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        clues = qa.get("clue", []) or []
        if isinstance(clues, str):
            clues = [clues]
        source_image: Path | None = None
        source_caption = ""
        for clue_round in clues:
            if clue_round in round_lookup:
                source_image, source_caption = round_lookup[clue_round]
                break
        if source_image is None and qa.get("question_image"):
            source_image = resolve_image_path(root, dataset, str(qa["question_image"]))
            source_caption = question
        if source_image is not None and source_image.exists() and question:
            queries.append(
                MemGalleryQuery(
                    query_id=f"{dataset}:qa_{index}",
                    dataset=dataset,
                    session_id=_clean_session_id(qa.get("session_id", f"qa{index}")),
                    question=question,
                    answer=answer,
                    source_image_path=source_image,
                    category=str(qa.get("point", "")),
                    source_caption=source_caption or question,
                )
            )
    return queries


def load_samples_from_data_dir(
    data_dir: str | Path,
    *,
    verify_hashes: bool = True,
    sample_ids: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> tuple[dict[str, Any], list[FrozenSample]]:
    dir_path = Path(data_dir).resolve()
    if not dir_path.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{dir_path}")

    subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    requested = set(sample_ids) if sample_ids is not None else None

    raw_samples: list[dict[str, Any]] = []
    for s_dir in subdirs:
        sid = s_dir.name
        if requested is not None and sid not in requested:
            continue
        meta_file = s_dir / "metadata.json"
        if meta_file.is_file():
            meta = load_json(meta_file)
        else:
            src_img = s_dir / "source.jpg"
            tgt_img = s_dir / "target.jpg"
            if not src_img.is_file() or not tgt_img.is_file():
                continue
            meta = {
                "sample_id": sid,
                "question": "",
                "answer": "",
                "source_image": str(src_img),
                "source_sha256": sha256_file(src_img),
                "target_image": str(tgt_img),
                "target_sha256": sha256_file(tgt_img),
                "target_text": "",
                "selection_score": 1.0,
                "phi_r": 0.0,
                "phi_contra": 0.0,
                "phi_c": 0.0,
            }
        raw_samples.append(meta)

    if requested is not None:
        found = {item["sample_id"] for item in raw_samples}
        missing = requested - found
        if missing:
            raise KeyError(f"数据目录中没有这些样本：{', '.join(sorted(missing))}")

    manifest_data = {
        "status": "frozen_for_execution",
        "data_dir": str(dir_path),
        "num_samples": len(raw_samples),
        "samples": raw_samples,
    }

    samples = [FrozenSample.from_mapping(item, base_dir=dir_path) for item in raw_samples]
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("数据目录中没有发现有效的样本。")
    if verify_hashes:
        verify_frozen_samples(samples)
    return manifest_data, samples


def load_frozen_manifest(
    path: str | Path,
    *,
    verify_hashes: bool = True,
    sample_ids: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> tuple[dict[str, Any], list[FrozenSample]]:
    target_path = Path(path).resolve()
    if target_path.is_dir():
        if (target_path / "manifest.json").is_file():
            target_path = target_path / "manifest.json"
        else:
            return load_samples_from_data_dir(
                target_path,
                verify_hashes=verify_hashes,
                sample_ids=sample_ids,
                max_samples=max_samples,
            )

    raw = load_json(target_path)
    if raw.get("status") not in {
        "frozen_from_saved_results_not_reselected_or_rerun",
        "frozen_for_execution",
    }:
        raise ValueError("历史清单状态不符合冻结参考要求。")
    requested = set(sample_ids) if sample_ids is not None else None
    base_dir = target_path.parent
    samples = [
        FrozenSample.from_mapping(item, base_dir=base_dir)
        for item in raw.get("samples", [])
        if requested is None or str(item.get("sample_id")) in requested
    ]
    sample_identifiers = [sample.sample_id for sample in samples]
    if len(set(sample_identifiers)) != len(sample_identifiers):
        raise ValueError("冻结清单包含重复的样本标识，无法安全创建独立输出目录。")
    if requested is not None:
        found = {sample.sample_id for sample in samples}
        missing = requested - found
        if missing:
            raise KeyError(f"历史清单中没有这些样本：{', '.join(sorted(missing))}")
    if not samples:
        raise ValueError("冻结清单没有可运行样本。")
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    if verify_hashes:
        verify_frozen_samples(samples)
    return raw, samples


def verify_frozen_samples(samples: Iterable[FrozenSample]) -> dict[str, int]:
    rows = list(samples)
    checked_paths: dict[Path, str] = {}
    for sample in rows:
        for path, expected in (
            (sample.source_image, sample.source_sha256),
            (sample.target_image, sample.target_sha256),
        ):
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"冻结清单引用图片不存在：{resolved}")
            actual = checked_paths.get(resolved)
            if actual is None:
                actual = sha256_file(resolved)
                checked_paths[resolved] = actual
            if actual != expected:
                raise ValueError(f"冻结清单引用图片已经改变：{resolved}")
    return {"rows": len(rows), "unique_images_checked": len(checked_paths)}
