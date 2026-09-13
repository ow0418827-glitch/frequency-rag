"""记忆后端统一数据格式及接口。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class MemoryEntry:
    entry_id: str
    text: str
    images: list[str]
    timestamp: str
    session_id: str
    image_ids: list[str] = field(default_factory=list)


@dataclass
class Retrieval:
    entries: list[MemoryEntry]
    candidate_ids: list[str]
    scores: list[float]
    route: str = "all"


class MemoryBackend(Protocol):
    def store(self, entry: MemoryEntry) -> None: ...
    def recall(self, question: str, images=()) -> Retrieval: ...
    def reset(self) -> None: ...
    def save(self, path) -> None: ...
    def restore(self, path) -> None: ...
