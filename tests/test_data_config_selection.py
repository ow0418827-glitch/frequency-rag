from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frequency_rag.config import load_config
from frequency_rag import data as data_module
from frequency_rag.data import (
    FrozenSample,
    MemGalleryCandidate,
    MemGalleryQuery,
    load_frozen_manifest,
    verify_frozen_samples,
)
from frequency_rag.io import load_json, save_json, sha256_file
from frequency_rag.selection import lexical_gate, select_target_for_query


def test_executable_config_loads_and_design_config_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.json")
    assert config.status == "implemented_not_benchmarked"
    assert len(config.surrogates) == 3
    with pytest.raises(ValueError, match="设计草案"):
        load_config(root / "configs" / "design_defaults.json")


def test_frozen_manifest_preserves_ten_rows_and_duplicate_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    _, samples = load_frozen_manifest(
        root / "manifests" / "reference_snapshot.json",
        verify_hashes=False,
    )
    assert len(samples) == 10
    assert len({sample.source_sha256 for sample in samples}) == 8
    assert len({sample.target_sha256 for sample in samples}) == 2


def test_frozen_manifest_sample_filter_reports_unknown_id() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(KeyError, match="missing"):
        load_frozen_manifest(
            root / "manifests" / "reference_snapshot.json",
            verify_hashes=False,
            sample_ids=["missing"],
        )


def test_frozen_manifest_rejects_duplicate_sample_identifiers(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_json(root / "manifests" / "reference_snapshot.json")
    manifest["samples"] = [manifest["samples"][0], dict(manifest["samples"][0])]
    duplicate = tmp_path / "duplicate.json"
    save_json(duplicate, manifest)
    with pytest.raises(ValueError, match="重复的样本标识"):
        load_frozen_manifest(duplicate, verify_hashes=False)


def test_frozen_image_hash_is_computed_once_per_unique_path(tmp_path, monkeypatch) -> None:
    image = tmp_path / "image.png"
    from PIL import Image

    Image.new("RGB", (2, 2), (10, 20, 30)).save(image)
    digest = sha256_file(image)
    sample = FrozenSample(
        sample_id="sample",
        question="q",
        answer="a",
        source_image=image,
        source_sha256=digest,
        target_image=image,
        target_sha256=digest,
        target_text="t",
        selection_score=0.0,
        phi_r=0.0,
        phi_contra=0.0,
        phi_c=0.0,
        raw={},
    )
    calls = 0
    original = data_module.sha256_file

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(data_module, "sha256_file", counted)
    result = verify_frozen_samples([sample, replace(sample, sample_id="sample-two")])
    assert result == {"rows": 2, "unique_images_checked": 1}
    assert calls == 1


def test_lexical_gate_matches_reference_edge_cases() -> None:
    assert lexical_gate("anything", "")
    assert not lexical_gate("a blue robot", "robot")
    assert lexical_gate("a blue robot", "cat")
    assert not lexical_gate("one two three four", "one two three four")
    assert lexical_gate("one two three", "one two three four")


def test_selection_uses_stable_order_and_penalty_not_hard_deletion(cpu_config, tmp_path) -> None:
    source = tmp_path / "source.png"
    source.touch()
    first = tmp_path / "first.png"
    first.touch()
    second = tmp_path / "second.png"
    second.touch()
    query = MemGalleryQuery(
        query_id="q",
        dataset="d",
        session_id="s",
        question="question",
        answer="answer",
        source_image_path=source,
        category="",
        source_caption="",
    )
    candidates = [
        MemGalleryCandidate("1", "d", "s", "1", first, "answer"),
        MemGalleryCandidate("2", "d", "s", "2", second, "answer"),
    ]
    image_embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    caption_embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    result = select_target_for_query(
        query,
        candidates,
        image_embeddings,
        caption_embeddings,
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
        cpu_config.selection,
    )
    assert result is not None
    assert result.target_candidate.candidate_id == "1"
    assert result.all_candidates_failed_lexical_gate
    assert result.score < -999_000


def test_config_rejects_silent_device_or_evaluator_behavior(cpu_config) -> None:
    with pytest.raises(ValueError, match="主评估器"):
        replace(
            cpu_config,
            evaluation=replace(cpu_config.evaluation, use_primary_evaluator_gradient=True),
        ).validate()
    with pytest.raises(ValueError, match="fp32"):
        replace(cpu_config, attack_precision="fp16").validate()
    with pytest.raises(ValueError, match="计时重复数"):
        replace(
            cpu_config,
            experiment=replace(cpu_config.experiment, timing_repetitions=0),
        ).validate()
