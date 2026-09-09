from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .config import SelectionConfig
from .data import MemGalleryCandidate, MemGalleryQuery


WORD_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class TargetSelectionResult:
    query: MemGalleryQuery
    target_candidate: MemGalleryCandidate
    score: float
    phi_r: float
    phi_contra: float
    phi_c: float
    target_text: str
    selected_candidate_passed_lexical_gate: bool
    all_candidates_failed_lexical_gate: bool


def normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norm, 1e-12, None)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def lexical_gate(candidate_caption: str, ground_truth_answer: str) -> bool:
    answer_tokens = tokenize(ground_truth_answer)
    if not answer_tokens:
        return True
    overlap = answer_tokens & tokenize(candidate_caption)
    if len(answer_tokens) <= 2:
        return not overlap
    return len(overlap) < len(answer_tokens) * 0.8


def select_target_for_query(
    query: MemGalleryQuery,
    candidates: list[MemGalleryCandidate],
    candidate_image_embeds: np.ndarray,
    candidate_caption_embeds: np.ndarray,
    query_text_embed: np.ndarray,
    ground_truth_embed: np.ndarray,
    config: SelectionConfig,
) -> TargetSelectionResult | None:
    if len(candidates) != len(candidate_image_embeds) or len(candidates) != len(candidate_caption_embeds):
        raise ValueError("候选条目与候选向量数量不一致。")
    query_embedding = normalize(query_text_embed)
    answer_embedding = normalize(ground_truth_embed)
    source_key = str(query.source_image_path.resolve())
    scores: list[tuple[float, float, float, float, bool, MemGalleryCandidate]] = []

    for index, candidate in enumerate(candidates):
        if str(candidate.image_path.resolve()) == source_key:
            continue
        image_embedding = normalize(candidate_image_embeds[index])
        caption_embedding = normalize(candidate_caption_embeds[index])
        image_query = float(np.dot(image_embedding, query_embedding))
        caption_query = float(np.dot(caption_embedding, query_embedding))
        phi_r = 0.5 * (image_query + caption_query)
        similarity_to_answer = float(np.dot(caption_embedding, answer_embedding))
        phi_contra = 1.0 - similarity_to_answer
        phi_c = max(0.0, similarity_to_answer)
        passed = lexical_gate(candidate.caption, query.answer)
        score = (
            config.retrieval_weight * phi_r
            + config.contradiction_weight * phi_contra
            - config.correct_alignment_weight * phi_c
        )
        if not passed:
            score -= config.lexical_penalty
        scores.append((score, phi_r, phi_contra, phi_c, passed, candidate))

    if not scores:
        return None
    # Python 的排序稳定；分数相同时保留候选池原顺序，与参考实现一致。
    scores.sort(key=lambda item: item[0], reverse=True)
    score, phi_r, phi_contra, phi_c, passed, candidate = scores[0]
    return TargetSelectionResult(
        query=query,
        target_candidate=candidate,
        score=float(score),
        phi_r=float(phi_r),
        phi_contra=float(phi_contra),
        phi_c=float(phi_c),
        target_text=candidate.caption or query.question,
        selected_candidate_passed_lexical_gate=passed,
        all_candidates_failed_lexical_gate=not any(item[4] for item in scores),
    )
