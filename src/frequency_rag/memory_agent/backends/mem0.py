"""真实事实记忆服务适配：描述图片、写入事实、检索及恢复。"""
from __future__ import annotations
from dataclasses import asdict
import inspect
import uuid
from frequency_rag.common.io import load_json, save_json
from frequency_rag.common.credentials import read_api_key
from .base import MemoryEntry, Retrieval

class Mem0Memory:
    """使用真实 mem0 客户端；独立用户命名空间，禁止清空共享服务。"""

    def __init__(self, config, llm, *, candidates=10, top_k=3, client=None):
        if not 0 < top_k <= candidates:
            raise ValueError("无效的检索数量。")
        if client is None:
            try:
                from mem0 import Memory
            except ImportError as exc:
                raise RuntimeError("事实记忆后端需要安装可选依赖 mem0ai。") from exc
            def resolve_env(value):
                if isinstance(value, list):
                    return [resolve_env(child) for child in value]
                if not isinstance(value, dict):
                    return value
                resolved = {k: resolve_env(v) for k, v in value.items() if k != "api_key_env"}
                if "api_key_env" in value or value.get("api_key"):
                    key = read_api_key(value)
                    if not key:
                        raise ValueError("事实记忆配置的密钥未设置，请填写 api_key 或对应环境变量。")
                    resolved["api_key"] = key
                return resolved
            client = Memory.from_config(resolve_env(config))
        self.client, self.llm, self.top_k, self.candidates = client, llm, top_k, candidates
        self.reset()

    def reset(self):
        self.user_id = "frequency-rag-" + uuid.uuid4().hex
        self.entries = {}

    def store(self, entry):
        if entry.entry_id in self.entries:
            raise ValueError("重复记忆轮次。")
        description = self.llm.describe(entry.images) if entry.images else ""
        observation = entry.text + "\n" + description
        result = self.client.add([{"role": "user", "content": observation}], user_id=self.user_id,
                                 metadata={"entry_id": entry.entry_id, "session_id": entry.session_id,
                                           "timestamp": entry.timestamp})
        if not isinstance(result, dict) or "results" not in result:
            raise ValueError("Mem0 写入返回值缺少 results。")
        self.entries[entry.entry_id] = entry

    def recall(self, question, images=()):
        query = question + ("\n" + self.llm.describe(images) if images else "")
        options = self._search_options(self.client.search, self.candidates)
        result = self.client.search(query, **options)
        rows = result["results"]
        entries, scores, candidate_ids = [], [], []
        for row in rows:
            eid = row.get("metadata", {}).get("entry_id")
            if eid not in self.entries:
                raise ValueError("Mem0 检索结果缺少可追溯的原始轮次。")
            original = self.entries[eid]
            candidate_ids.append(eid)
            entries.append(MemoryEntry(eid, row["memory"], original.images, original.timestamp,
                                       original.session_id, original.image_ids))
            scores.append(float(row.get("score", 0)))
        return Retrieval(entries[:self.top_k], candidate_ids, scores[:self.top_k], "facts")

    def _search_options(self, method, limit):
        # 新版使用 filters/top_k；旧版明确支持 user_id/limit。避免 **kwargs 静默忽略数量。
        parameters = inspect.signature(method).parameters
        if "top_k" in parameters:
            return {"filters": {"user_id": self.user_id}, "top_k": limit}
        return {"user_id": self.user_id, "limit": limit}

    def save(self, path):
        facts = self.client.get_all(**self._search_options(self.client.get_all, 1000000))
        if len(facts["results"]) >= 1000000:
            raise ValueError("事实快照达到上限，拒绝保存可能被截断的记录。")
        save_json(path, {"kind": "mem0", "user_id": self.user_id,
                         "entries": [asdict(e) for e in self.entries.values()],
                         "facts": facts})

    def restore(self, path):
        data = load_json(path)
        if data["kind"] != "mem0":
            raise ValueError("不是事实记忆快照。")
        # 数据库本身须持久化；该清单恢复命名空间和原图映射，不重写数据库。
        self.user_id = data["user_id"]
        self.entries = {e["entry_id"]: MemoryEntry(**e) for e in data["entries"]}
