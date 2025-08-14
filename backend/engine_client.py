import requests
from typing import List, Dict, Any, Optional
from backend.config import (
    CORP_API_KEY,
    CORP_EMBED_BASE_URL,
    HTTP_TIMEOUT,
)

class EngineClient:
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: float = HTTP_TIMEOUT):
        self.api_key = api_key or CORP_API_KEY
        self.base_url = (base_url or CORP_EMBED_BASE_URL).rstrip("/")
        self.timeout = timeout
        if not self.api_key or not self.base_url:
            raise RuntimeError("EngineClient missing API key or base URL.")

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def create_embeddings(self, model: str, inputs: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": model, "input": inputs}
        r = requests.post(url, headers=self._headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return [item["embedding"] for item in data["data"]]

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": temperature}
        r = requests.post(url, headers=self._headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]