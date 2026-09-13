import copy
import json
import sys
from types import SimpleNamespace
import urllib.request

import pytest

from frequency_rag.common.credentials import read_api_key, redact_config
from frequency_rag.memory_agent.llm import ChatModel
from frequency_rag.memory_agent.backends.mem0 import Mem0Memory


@pytest.mark.parametrize("direct,expected", [("local-test-key", "local-test-key"), ("", "env-test-key"), ("  ", "env-test-key")])
def test_config_key_precedence_and_actual_request(monkeypatch, direct, expected):
    monkeypatch.setenv("TEST_CREDENTIAL_KEY", "env-test-key")
    config = {"model": "test-model", "base_url": "https://example.invalid/v1",
              "api_key": direct, "api_key_env": "TEST_CREDENTIAL_KEY"}
    original = copy.deepcopy(config)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    def request(req, **kwargs):
        assert req.get_header("Authorization") == "Bearer " + expected
        assert expected not in req.data.decode()
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", request)
    model = ChatModel(config)
    assert model.complete("hello") == "ok"
    assert expected not in json.dumps(model.config)
    assert config == original


def test_recursive_redaction_does_not_modify_input():
    config = {"nested": [{"API_KEY": "key", "Authorization": "bearer", "password": "pass"}],
              "api_key_env": "TEST_KEY", "max_tokens": 10}
    original = copy.deepcopy(config)
    safe = redact_config(config)
    assert safe["nested"][0] == {"API_KEY": "[REDACTED]", "Authorization": "[REDACTED]", "password": "[REDACTED]"}
    assert safe["api_key_env"] == "TEST_KEY" and safe["max_tokens"] == 10
    assert config == original
    with pytest.raises(ValueError, match="脱敏"):
        read_api_key({"api_key": "[REDACTED]"})


def test_mem0_resolves_local_and_environment_keys_without_network(monkeypatch):
    monkeypatch.setenv("TEST_CREDENTIAL_KEY", "env-test-key")
    captured = {}
    def from_config(config):
        captured.update(config)
        return object()
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=SimpleNamespace(from_config=from_config)))
    config = {"llm": {"config": {"api_key": "local-test-key", "api_key_env": "TEST_CREDENTIAL_KEY"}},
              "embedder": {"config": {"api_key": "", "api_key_env": "TEST_CREDENTIAL_KEY"}}}
    original = copy.deepcopy(config)
    Mem0Memory(config, object())
    assert captured["llm"]["config"] == {"api_key": "local-test-key"}
    assert captured["embedder"]["config"] == {"api_key": "env-test-key"}
    assert config == original
