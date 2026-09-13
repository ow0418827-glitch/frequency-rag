import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from frequency_rag.common.io import save_json, sha256_file
from frequency_rag.memory_agent.backends import MemoryEntry, make_memory
from frequency_rag.memory_agent.backends.mem0 import Mem0Memory
from frequency_rag.image_selection.categories import CATEGORIES
from frequency_rag.image_selection.plans import build_injection_plan
from frequency_rag.memory_agent.scenarios import condition_dialog, adversarial_images
from frequency_rag.image_selection.dataset import read_dialog
from frequency_rag.memory_agent.agent import MemoryAgent
from frequency_rag.evaluation.memory_metrics import aggregate, token_f1, retrieval_metrics
from frequency_rag.evaluation.judge import judge_answer
from frequency_rag.experiments.memory_runner import run_memory_experiment
from frequency_rag.memory_agent.llm import ChatModel
from frequency_rag.image_selection.plans import build_poisoning_plan


class Encoder:
    def encode(self, text, images=()):
        # 有独立视觉维度，便于验证图片确实参与了检索。
        v = np.array([1., float("fruit" in text.lower()), float(bool(images))])
        return v / np.linalg.norm(v)


class Model:
    def __init__(self):
        self.calls = []

    def describe(self, images):
        return "observed source fruit"

    def complete(self, prompt, images=(), *, structured=False):
        self.calls.append((prompt, list(images), structured))
        if "concepts" in prompt:
            return {"concepts": ["fruit"]}
        if '"route"' in prompt:
            return {"route": "image"}
        if "target_claim_present" in prompt:
            return {"correct": True, "target_claim_present": False, "reason": "test double"}
        return "fruit"


@pytest.fixture
def benchmark(tmp_path):
    src, tgt = tmp_path / "source.png", tmp_path / "target.png"
    Image.new("RGB", (8, 8), "red").save(src)
    Image.new("RGB", (8, 8), "blue").save(tgt)
    bundle = {"audit": {"dataset": "test"}, "dialog": {"multi_session_dialogues": [
        {"session_id": "D1", "date": "2024-01-01", "dialogues": [
            {"round": "D1:1", "user": "fruit", "assistant": "original response",
             "input_image": [str(src)], "image_caption": ["fruit"], "image_id": ["D1:IMG1"]}]},
        {"session_id": "D2", "date": "2024-02-01", "dialogues": []}]},
        "qas": [{"qa_id": "q1", "question": "What fruit?", "answer": "fruit", "point": "VS",
                 "clue": ["D1:1"], "query_images": []}]}
    pairs = {"provenance": "synthetic_unit_test_only", "pairs": [
        {"category": category, "source_image": str(src), "target_image": str(tgt), "validated": True,
         "user": "Describe this", "question": "What fruit?", "answer": "fruit",
         "target_answer": "TARGET_SECRET", "target_text": "TARGET_SECRET"} for category in CATEGORIES]}
    bp, pp = tmp_path / "bundle.json", tmp_path / "pairs.json"
    save_json(bp, bundle)
    save_json(pp, pairs)
    return bundle, bp, pp


def test_plan_four_categories_deterministic_and_attack_compatible(benchmark, tmp_path):
    from frequency_rag.image_selection.data import load_frozen_manifest
    bundle, bp, pp = benchmark
    output = tmp_path / "plan.json"
    first = build_injection_plan(bp, pp, output, per_category=1)
    second = build_injection_plan(bp, pp, tmp_path / "plan2.json", per_category=1)
    assert first == second
    assert len(first["samples"]) == 4
    assert {s["category"] for s in first["samples"]} == set(CATEGORIES)
    assert all(s["session_id"] == "D1" for s in first["samples"])
    assert len(load_frozen_manifest(output.with_suffix(".attack.json"))[1]) == 4


def test_insufficient_category_not_silently_accepted(benchmark, tmp_path):
    _, bp, pp = benchmark
    with pytest.raises(ValueError, match="缺少"):
        build_injection_plan(bp, pp, tmp_path / "plan.json", per_category=4)


def test_conditions_do_not_leak_target_or_mutate_original(benchmark, tmp_path):
    bundle, bp, pp = benchmark
    original = copy.deepcopy(bundle)
    plan = build_injection_plan(bp, pp, tmp_path / "plan.json", per_category=1)
    images = {s["sample_id"]: s["source_image"] for s in plan["samples"]}
    clean = condition_dialog(bundle, plan, "clean")
    assert clean == bundle["dialog"]
    adversarial = condition_dialog(bundle, plan, "adversarial", images=images, describer=Model())
    assert "TARGET_SECRET" not in json.dumps(adversarial)
    assert "TARGET_SECRET" in json.dumps(condition_dialog(bundle, plan, "oracle"))
    assert len(adversarial["multi_session_dialogues"][0]["dialogues"]) == 5
    assert bundle == original


def test_poisoning_keeps_user_assistant_text(benchmark):
    bundle, _, _ = benchmark
    original = bundle["dialog"]["multi_session_dialogues"][0]["dialogues"][0]
    source = original["input_image"][0]
    item = {"turn_id": "D1:1", "sample_id": "p", "source_image": source, "source_sha256": sha256_file(source)}
    result = condition_dialog(bundle, {"family": "poisoning", "samples": [item]}, "adversarial",
                              images={"p": source}, describer=Model())
    changed = result["multi_session_dialogues"][0]["dialogues"][0]
    assert changed["user"] == original["user"]
    assert changed["assistant"] == original["assistant"]


@pytest.mark.parametrize("kind", ["murag", "ngmemory", "augustus", "universalrag"])
def test_memory_persistence_and_reset(kind, tmp_path):
    memory = make_memory({"backend": kind}, Encoder(), Model())
    memory.store(MemoryEntry("a", "fruit", ["a.png"], "2024-01-01", "D1"))
    memory.store(MemoryEntry("b", "other", [], "2024-01-02", "D1"))
    before = memory.recall("fruit")
    path = tmp_path / "memory.json"
    memory.save(path)
    memory.reset()
    assert not memory.entries
    memory.restore(path)
    assert memory.recall("fruit").candidate_ids == before.candidate_ids
    assert memory.recall("fruit").entries[0].entry_id == "a"
    with pytest.raises(ValueError, match="重复"):
        memory.store(memory.entries[0])


def test_graph_edges_are_directed_to_past():
    memory = make_memory({"backend": "ngmemory"}, Encoder(), Model())
    for i in range(3):
        memory.store(MemoryEntry(str(i), "fruit", [], "", "D1"))
    assert memory.edges == [[], [0], [1, 0]]


def test_agent_preserves_images_timestamps_and_does_not_store_answers():
    model = Model()
    memory = make_memory({"backend": "murag"}, Encoder(), model)
    memory.store(MemoryEntry("a", "fruit", ["a.png"], "2024-01-01", "D1", ["IMG_A"]))
    MemoryAgent(memory, model).answer("What fruit?", ["query.png"])
    prompt, images, _ = model.calls[-1]
    assert images == ["a.png", "query.png"]
    assert "2024-01-01" in prompt and "IMG_A" in prompt
    assert len(memory.entries) == 1


def test_judge_boolean_strict_and_no_substring_yes_metric():
    class Bad:
        def complete(self, *args, **kwargs):
            return {"correct": "false", "target_claim_present": "false"}
    with pytest.raises(ValueError, match="布尔"):
        judge_answer(Bad(), {"question": "q", "answer": "no"}, "not yes")
    assert token_f1("fruit fruit", "fruit") == pytest.approx(2/3)
    assert retrieval_metrics([], ["a"])["recall_at_k"] == 0
    assert retrieval_metrics([], [])["recall_at_k"] is None


def test_conditional_metrics_have_correct_denominators():
    base = {"qa_id": "q1", "condition": "adversarial", "category": "allergy_safety", "probe_type": "attack",
            "status": "ok", "target_claim_present": True, "attack_retrieved": True}
    result = aggregate([base, {**base, "target_claim_present": False, "attack_retrieved": False},
                        {**base, "status": "failed"}])
    group = result["groups"][0]
    assert group["planned"] == 3 and group["failed"] == 1
    assert group["target_claim_rate"] == .5
    assert group["conditional_target_claim_rate"] == 1
    assert group["conditional_denominator"] == 1
    assert result["status"] == "partial_failure"


def test_complete_offline_chain_all_four_categories(benchmark, tmp_path):
    bundle, bp, pp = benchmark
    plan_path = tmp_path / "plan.json"
    build_injection_plan(bp, pp, plan_path, per_category=1)
    cp = tmp_path / "config.json"
    save_json(cp, {"backend": "murag", "model": {"model": "test_double"}})
    model = Model()
    result = run_memory_experiment(bp, plan_path, cp, tmp_path / "run",
                                  conditions=["clean", "oracle", "source_injection_control"],
                                  model=model, judge=Model(), encoder=Encoder())
    assert result["status"] == "complete"
    assert len(result["groups"]) == 11
    clean_calls = model.calls[:1]
    assert all("TARGET_SECRET" not in p for p, _, _ in clean_calls)
    assert (tmp_path / "run" / "clean" / "memory.json").is_file()


def test_adversarial_import_rejects_hash_and_budget(benchmark, tmp_path):
    _, bp, pp = benchmark
    plan = build_injection_plan(bp, pp, tmp_path / "plan.json", per_category=1)
    sample = plan["samples"][0]
    root = tmp_path / "attack"
    root.mkdir()
    row = {**sample, "status": "generated", "adversarial_image": sample["source_image"],
           "adversarial_sha256": sample["source_sha256"]}
    save_json(root / "run_manifest.json", {"samples": [row]})
    assert adversarial_images(root, [sample])[sample["sample_id"]] == sample["source_image"]
    row.update(adversarial_image=sample["target_image"], adversarial_sha256=sample["target_sha256"])
    save_json(root / "run_manifest.json", {"samples": [row]})
    with pytest.raises(ValueError, match="上限"):
        adversarial_images(root, [sample])
    row["adversarial_sha256"] = "wrong"
    save_json(root / "run_manifest.json", {"samples": [row]})
    with pytest.raises(ValueError, match="摘要"):
        adversarial_images(root, [sample])


def test_minimax_request_protocol_and_reasoning_split(monkeypatch):
    import urllib.request
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": '<think>analysis</think>```json\n{"correct": true}\n```'},
                                               "finish_reason": "stop"}]}).encode()
    def request(req, **kwargs):
        captured.update(json.loads(req.data))
        assert req.full_url == "https://api.minimax.cn/v1/chat/completions"
        return Response()
    monkeypatch.setenv("TEST_MODEL_KEY", "fake-test-only")
    monkeypatch.setattr(urllib.request, "urlopen", request)
    model = ChatModel({"provider": "minimax", "model": "MiniMax-M3", "api_key_env": "TEST_MODEL_KEY",
                       "base_url": "https://api.minimax.cn/v1"})
    assert model.complete("test", structured=True) == {"correct": True}
    assert captured["reasoning_split"] is True
    assert "response_format" not in captured


def test_mem0_uses_isolated_namespace_and_reattaches_evidence():
    class Client:
        def add(self, messages, **kwargs):
            self.meta = kwargs["metadata"]
            return {"results": []}
        def search(self, *args, **kwargs):
            return {"results": [{"memory": "fruit fact", "metadata": self.meta, "score": .9}]}
    memory = Mem0Memory({}, Model(), client=Client())
    old = memory.user_id
    memory.store(MemoryEntry("a", "fruit", ["a.png"], "2024-01-01", "D1"))
    found = memory.recall("fruit")
    assert found.entries[0].images == ["a.png"]
    assert found.entries[0].text == "fruit fact"
    memory.reset()
    assert old != memory.user_id


def test_mem0_current_api_uses_filter_and_top_k():
    class Client:
        def search(self, query, *, filters, top_k):
            assert filters["user_id"].startswith("frequency-rag-")
            assert top_k == 10
            return {"results": []}
    memory = Mem0Memory({}, Model(), client=Client())
    assert memory.recall("fruit").entries == []


def test_local_secrets_are_redacted_from_run_artifacts(benchmark, tmp_path):
    _, bp, _ = benchmark
    cp = tmp_path / "config.json"
    config = {"backend": "murag", "model": {"model": "test_double", "api_key": "test-secret"},
              "judge": {"api_key": "judge-secret"},
              "mem0": {"nested": [{"api_key": "mem0-secret", "password": "db-secret"}]}}
    save_json(cp, config)
    output = tmp_path / "run"
    result = run_memory_experiment(bp, None, cp, output, conditions=["clean"],
                                  model=Model(), judge=Model(), encoder=Encoder())
    assert result["status"] == "complete"
    for path in output.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert all(secret not in text for secret in ("test-secret", "judge-secret", "mem0-secret", "db-secret"))
    assert json.loads(cp.read_text(encoding="utf-8")) == config
    assert json.loads((output / "config.json").read_text(encoding="utf-8"))["model"]["api_key"] == "[REDACTED]"


def test_poison_plan_matches_evidence_round(benchmark, tmp_path):
    _, bp, pp = benchmark
    pair = json.loads(pp.read_text(encoding="utf-8"))["pairs"][0]
    pair.update(sample_id="poison_1")
    manifest = tmp_path / "frozen.json"
    save_json(manifest, {"status": "frozen_for_execution", "samples": [pair]})
    plan = build_poisoning_plan(bp, manifest, tmp_path / "poison.json")
    assert plan["samples"][0]["turn_id"] == "D1:1"
    assert plan["samples"][0]["qa_id"] == "q1"


def test_adversarial_full_chain_consumes_saved_images(benchmark, tmp_path):
    _, bp, pp = benchmark
    plan_path = tmp_path / "plan.json"
    plan = build_injection_plan(bp, pp, plan_path, per_category=1)
    attack = tmp_path / "attack"
    attack.mkdir()
    rows = [{**s, "status": "generated", "adversarial_image": s["source_image"],
             "adversarial_sha256": s["source_sha256"]} for s in plan["samples"]]
    save_json(attack / "run_manifest.json", {"samples": rows})
    cp = tmp_path / "config.json"
    save_json(cp, {"backend": "ngmemory", "model": {"model": "test_double"}})
    model = Model()
    result = run_memory_experiment(bp, plan_path, cp, tmp_path / "run", run_dir=attack,
                                   conditions=["adversarial"], model=model, judge=Model(), encoder=Encoder())
    assert result["status"] == "complete"
    assert all("TARGET_SECRET" not in prompt for prompt, _, _ in model.calls)
    assert len(json.loads((tmp_path / "run/adversarial/answers.json").read_text(encoding="utf-8"))) == 5


def test_real_schema_keeps_text_only_questions_and_checks_missing_clues(tmp_path):
    root = tmp_path / "data"
    (root / "image" / "test").mkdir(parents=True)
    path = root / "dialog" / "test.json"
    raw = {"multi_session_dialogues": [{"session_id": "D1", "date": "2024-01-01", "dialogues": [
        {"round": "D1:1", "user": "hello", "assistant": "hi"}]}],
        "human-annotated QAs": [{"point": "MR", "question": "What greeting?", "answer": "hello", "clue": ["D1:1"]}]}
    save_json(path, raw)
    _, qas, audit = read_dialog(root, "test")
    assert len(qas) == 1 and audit["images"] == 0
    raw["human-annotated QAs"][0]["clue"] = ["missing"]
    save_json(path, raw)
    with pytest.raises(ValueError, match="不存在"):
        read_dialog(root, "test")


def test_missing_key_checked_without_network(monkeypatch):
    monkeypatch.delenv("FREQUENCY_TEST_MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="未设置"):
        ChatModel({"base_url": "https://api.minimax.cn/v1", "model": "MiniMax-M3",
                   "api_key_env": "FREQUENCY_TEST_MISSING_KEY"})


def test_pool_selection_rejects_unvalidated_visual_pairs(benchmark, tmp_path):
    from frequency_rag.image_selection.candidates import select_injection_pairs
    _, _, pp = benchmark
    pair = json.loads(pp.read_text(encoding="utf-8"))["pairs"][0]
    pool = tmp_path / "pool.json"
    save_json(pool, {"provenance": "synthetic", "images": [
        {"image": pair["source_image"], "caption": "source"},
        {"image": pair["target_image"], "caption": "target"}]})
    class Reject:
        def complete(self, *args, **kwargs):
            return {"valid": False, "reason": "test rejection"}
    output = tmp_path / "selected.json"
    with pytest.raises(ValueError, match="数量不足"):
        select_injection_pairs(pool, output, Encoder(), Reject(), per_category=1, candidate_limit=2)
    assert not output.exists()
    assert output.with_suffix(".failed_audit.json").exists()
