"""本地配置密钥读取及配置快照脱敏。"""
from __future__ import annotations

import os


def read_api_key(config, *, default_env=None):
    """显式非空密钥优先，空值回退到指定环境变量。"""
    key = config.get("api_key")
    if key is not None and not isinstance(key, str):
        raise ValueError("模型密钥必须是字符串。")
    if key and key.strip():
        if key.strip() == "[REDACTED]":
            raise ValueError("脱敏运行快照不能作为密钥配置，请使用本地配置文件。")
        return key.strip()
    name = config.get("api_key_env") or default_env
    return os.environ.get(name, "").strip() if name else ""


def redact_config(value):
    """递归复制配置，隐藏凭据字段，保留环境变量名等复现参数。"""
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if key.lower() in {
            "api_key", "authorization", "password", "token", "access_token", "secret"
        } and child else redact_config(child)) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_config(child) for child in value]
    return value
