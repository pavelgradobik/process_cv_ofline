from typing import List, Sequence, Optional
import time
import requests
from backend.config import (
    CORP_EMBED_BASE_URL, CORP_API_KEY, CORP_EMBED_PROVIDER, CORP_EMBED_MODEL,
    HTTP_TIMEOUT, VERIFY_SSL, RETRY_TIMES, EMBED_BATCH,
)

class CorporateEmbeddingClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = HTTP_TIMEOUT,
        verify_ssl: bool = VERIFY_SSL,
        retries: int = RETRY_TIMES,
        batch_size: int = EMBED_BATCH,
    ):
        self.base = (base_url or CORP_EMBED_BASE_URL).rstrip("/")
        self.key = api_key or CORP_API_KEY
        self.provider = provider or CORP_EMBED_PROVIDER
        self.model = model or CORP_EMBED_MODEL
        self.timeout = timeout
        self.verify = verify_ssl
        self.retries = max(0, retries)
        self.batch_size = max(1, batch_size)

        if not self.key:
            raise RuntimeError("CORP_API_KEY is not set.")
        if not self.base:
            raise RuntimeError("CORP_EMBED_BASE_URL is not set.")

        self._url = f"{self.base}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        self._session = requests.Session()

    def _post_once(self, payload: dict) -> List[List[float]]:
        r = self._session.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self.timeout,
            verify=self.verify,
        )
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return [row.get("embedding") for row in data["data"]]

        if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
            return data

        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]

        raise ValueError(f"Unexpected embeddings response shape: {str(data)[:400]}")

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i:i + self.batch_size])
            payload = {
                "provider": self.provider,
                "model": self.model,
                "input": chunk,
            }
            attempt = 0
            backoff = 0.5
            while True:
                try:
                    vecs = self._post_once(payload)
                    if len(vecs) != len(chunk):
                        raise ValueError(f"Embeddings count mismatch: got {len(vecs)} for {len(chunk)} inputs")
                    out.extend(vecs)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > self.retries:
                        raise
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
        return out

    def embed_query(self, text: str) -> List[float]:
        vecs = self.embed_texts([text])
        return vecs[0]