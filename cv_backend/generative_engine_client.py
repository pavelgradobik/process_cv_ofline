from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import requests

from cv_backend.config import (
    GENERATIVE_ENGINE_BASE_URL,
    GENERATIVE_ENGINE_API_KEY,
    GENERATIVE_ENGINE_MODEL,
    REQUEST_TIMEOUT,
)

__VERSION__ = "corp_http_client/1.0.0"

def _endpoint(path: str) -> str:
    base = GENERATIVE_ENGINE_BASE_URL.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if GENERATIVE_ENGINE_API_KEY:
        h["Authorization"] = f"Bearer {GENERATIVE_ENGINE_API_KEY}"
    return h


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any] | str:
    resp = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    text = resp.text
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return text


def _extract_text(response: Dict[str, Any] | str) -> str:
    if isinstance(response, str):
        return response

    try:
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                if "text" in ch0 and isinstance(ch0["text"], str):
                    return ch0["text"]
    except Exception:
        pass

    for key in ("text", "output", "answer", "data"):
        v = response.get(key)
        if isinstance(v, str):
            return v

    try:
        return json.dumps(response, ensure_ascii=False, indent=2)
    except Exception:
        return str(response)


class Client:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or GENERATIVE_ENGINE_BASE_URL).rstrip("/")
        self.api_key = api_key or GENERATIVE_ENGINE_API_KEY
        self.model = model or GENERATIVE_ENGINE_MODEL
        self.timeout = float(timeout or REQUEST_TIMEOUT)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **extra,
    ) -> str:
        url = _endpoint("chat/completions")
        payload: Dict[str, Any] = {
            "messages": messages,
            "model": model or self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if extra:
            payload.update(extra)
        resp = _post_json(url, payload, timeout=self.timeout)
        return _extract_text(resp)

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "assistant", "content": prompt}]
        return self.chat(messages=messages, **kwargs)

    def complete(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def predict(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


def get_client() -> Client:
    return Client()

CorporateClient = Client
CorporateEngine = Client
EngineClient = Client
GenerativeClient = Client
client = Client