"""向量存储、持久化及有界图遍历；具体策略由独立后端提供。"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from frequency_rag.common.numerics import normalize
from frequency_rag.common.io import load_json, save_json
from .base import MemoryEntry, Retrieval


class VectorMemory:
    kind = "abstract"
    graph_enabled = False
    edge_threshold = 0.6
    traversal_threshold = 0.4

    def __init__(self, encoder, llm, *, candidates=10, top_k=3):
        if self.kind == "abstract":
            raise TypeError("请通过注册表选择具体记忆后端。")
        if not 0 < top_k <= candidates:
            raise ValueError("要求 0 < top_k <= candidates。")
        self.encoder, self.llm = encoder, llm
        self.candidates, self.top_k = candidates, top_k
        self.reset()

    def reset(self):
        self.entries, self.vectors, self.tags, self.edges = [], [], [], []

    def extract_tags(self, text):
        return set()

    def link_allowed(self, tags, previous_tags):
        return True

    def route_query(self, question, images):
        return "all"

    def rank_scores(self, question, cosines):
        return cosines.copy()

    def store(self, entry: MemoryEntry):
        if any(e.entry_id == entry.entry_id for e in self.entries):
            raise ValueError(f"重复记忆标识：{entry.entry_id}")
        vector = normalize(np.asarray(self.encoder.encode(entry.text, entry.images), dtype=np.float32))
        if vector.ndim != 1 or not np.all(np.isfinite(vector)) or not np.linalg.norm(vector):
            raise ValueError("无效记忆向量。")
        tags = self.extract_tags(entry.text)
        neighbors = []
        if self.graph_enabled:
            for i, old in enumerate(self.vectors):
                similarity = float(old @ vector)
                if similarity >= self.edge_threshold and self.link_allowed(tags, self.tags[i]):
                    neighbors.append((similarity, i))
        self.edges.append([i for _, i in sorted(neighbors, reverse=True)[:10]])
        self.entries.append(entry)
        self.vectors.append(vector)
        self.tags.append(tags)

    def recall(self, question, images=()):
        route = self.route_query(question, images)
        if route == "no" or not self.entries:
            return Retrieval([], [], [], route)
        vector = self.encoder.encode(question, images)
        cosines = np.asarray(self.vectors) @ vector
        scores = self.rank_scores(question, cosines)
        eligible = [i for i, e in enumerate(self.entries) if route != "image" or e.images]
        ranked = sorted(eligible, key=lambda i: (-float(scores[i]), i))
        seeds = ranked[:self.top_k] if self.graph_enabled else ranked[:self.candidates]
        collected, seen = [], set()

        def visit(i, depth):
            if i in seen or len(collected) >= self.candidates:
                return
            seen.add(i)
            collected.append(i)
            if depth < 3:
                for neighbor in self.edges[i]:
                    if cosines[neighbor] >= self.traversal_threshold:
                        visit(neighbor, depth + 1)

        for i in seeds:
            visit(i, 0)
        collected.sort(key=lambda i: (-float(scores[i]), i))
        selected = collected[:self.top_k]
        return Retrieval([self.entries[i] for i in selected],
                         [self.entries[i].entry_id for i in collected],
                         [float(scores[i]) for i in selected], route)

    def save(self, path):
        save_json(path, {"kind": self.kind, "candidates": self.candidates, "top_k": self.top_k,
                         "entries": [asdict(e) for e in self.entries],
                         "vectors": [v.tolist() for v in self.vectors],
                         "tags": [sorted(t) for t in self.tags], "edges": self.edges})

    def restore(self, path):
        data = load_json(path)
        if (data["kind"], data["candidates"], data["top_k"]) != (self.kind, self.candidates, self.top_k):
            raise ValueError("记忆快照架构或检索设置不一致。")
        self.entries = [MemoryEntry(**e) for e in data["entries"]]
        self.vectors = [np.asarray(v, dtype=np.float32) for v in data["vectors"]]
        self.tags, self.edges = [set(t) for t in data["tags"]], data["edges"]
