"""验证目录迁移后的加载、可扩展后端及递归源码追溯。"""
from types import SimpleNamespace

import pytest

from frequency_rag.common.paths import PROJECT_ROOT
from frequency_rag.common.provenance import implementation_snapshot
from frequency_rag.memory_agent.backends import registry


@pytest.mark.parametrize("name,class_name", [
    ("murag", "MuRAGMemory"), ("ngmemory", "NGMemory"),
    ("augustus", "AugustusMemory"), ("universalrag", "UniversalRAGMemory"),
])
def test_registry_selects_distinct_implementation(name, class_name):
    memory = registry.make_memory({"backend": name, "top_k": 2, "candidates": 5}, None, None)
    assert type(memory).__name__ == class_name
    assert memory.kind == name
    assert memory.top_k == 2 and memory.candidates == 5
    assert memory.recall if hasattr(memory, "recall") else False


def test_fifth_backend_is_loaded_only_when_selected(monkeypatch):
    calls = []
    class FakeMem0:
        def __init__(self, config, llm, **options):
            self.config, self.options = config, options
    def loader(module):
        calls.append(module)
        return SimpleNamespace(Mem0Memory=FakeMem0)
    monkeypatch.setattr(registry, "import_module", loader)
    memory = registry.make_memory({"backend": "mem0", "mem0": {"test": True}}, None, None)
    assert calls == ["frequency_rag.memory_agent.backends.mem0"]
    assert memory.config == {"test": True}
    assert memory.options == {"candidates": 10, "top_k": 3}


def test_custom_backend_and_duplicate_registration(monkeypatch):
    monkeypatch.setattr(registry, "_custom_factories", {})
    marker = object()
    registry.register_backend("test_backend", lambda config, encoder, llm: marker)
    assert registry.make_memory({"backend": "test_backend"}, None, None) is marker
    assert "test_backend" in registry.available_backends()
    with pytest.raises(ValueError, match="已注册"):
        registry.register_backend("murag", lambda *args: marker)
    with pytest.raises(ValueError, match="未知"):
        registry.make_memory({"backend": "missing"}, None, None)


def test_snapshot_covers_nested_implementations_and_distinguishes_paths():
    assert (PROJECT_ROOT / "configs/default.json").is_file()
    snapshot = implementation_snapshot()
    paths = [item["path"] for item in snapshot["files"]]
    assert len(paths) == len(set(paths))
    for module in ["pixel_attack/update.py", "frequency_attack/dct.py",
                   "image_selection/candidates.py", "memory_agent/llm.py",
                   "memory_agent/backends/mem0.py", "evaluation/judge.py",
                   "experiments/memory_runner.py"]:
        assert "src/frequency_rag/" + module in paths
    assert snapshot["content_identity"]
