"""模型服务及记忆编码；真实模型与离线测试替身由调用者显式注入。"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
import urllib.error
import urllib.request


class ChatModel:
    """兼容聊天补全协议，密钥只从环境变量读取，不写入运行记录。"""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.model = config["model"]
        if not self.model:
            raise ValueError("模型名称不能为空。")
        self.base_url = config["base_url"].rstrip("/")
        self.key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")
        if config.get("require_api_key", True) and not self.key:
            raise ValueError("模型服务密钥环境变量未设置。")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("模型服务地址必须使用 HTTP 或 HTTPS。")

    def complete(self, prompt: str, images=(), *, structured=False):
        if images and not self.config.get("supports_vision", True):
            raise ValueError("所配置模型不支持图像输入，请为视觉描述／回答配置多模态模型。")
        content = [{"type": "text", "text": prompt}]
        for path in images:
            path = Path(path)
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": self.config.get("temperature", 0),
                   "max_tokens": self.config.get("max_tokens", 1024)}
        if self.config.get("provider") == "minimax":
            payload["reasoning_split"] = True
        if structured and self.config.get("json_mode", False):
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.config.get("timeout", 120)) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # 响应体可能包含服务端回显的凭据或输入，避免直接记录。
            raise RuntimeError(f"模型服务请求失败，状态码 {exc.code}") from None
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("模型响应被长度上限截断。")
        answer = choice["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("模型服务返回空响应。")
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        if not answer:
            raise ValueError("模型响应没有最终答案。")
        if structured:
            answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer).strip()
            result = json.loads(answer)
            if not isinstance(result, dict):
                raise ValueError("模型结构化响应必须是对象。")
            return result
        return answer

    def describe(self, images):
        return self.complete("Describe these images faithfully, including visible objects, names, text and layout. "
                             "Do not infer invisible attributes. Return a plain factual description.", images)
