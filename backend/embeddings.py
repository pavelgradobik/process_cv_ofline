from typing import Optional, Callable, List
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.config import CORP_EMBED_MODEL, EMBED_BATCH
from backend.engine_client import EngineClient

ProgressCb = Optional[Callable[[int, int, str], None]]

class OfflineEmbedder:
    def __init__(self, dimension: int = 384):
        self.vectorizer = TfidfVectorizer(max_features=dimension)
        self.fitted = False
        self.matrix: Optional[spmatrix] = None

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before embed()")
        vec_sparse = self.vectorizer.transform([text])
        vec = np.asarray(vec_sparse.todense()).flatten()
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before embed_batch()")
        mat = self.vectorizer.transform(texts)
        return np.asarray(mat.todense())

    def embed_batch_with_progress(self, texts: List[str], update: ProgressCb = None) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before embed_batch_with_progress()")
        total = len(texts)
        rows = []
        for i, t in enumerate(texts, 1):
            vec_sparse = self.vectorizer.transform([t])
            vec = np.asarray(vec_sparse.todense()).flatten()
            rows.append(vec)
            if update and (i % 200 == 0 or i == total):
                update(i, total, "Vectorizing")
        return np.vstack(rows)

class EngineEmbedder:
    def __init__(self, model: str = CORP_EMBED_MODEL, batch: int = EMBED_BATCH):
        self.client = EngineClient()
        self.model = model
        self.batch = max(1, int(batch))
        self.fitted = True

    def fit(self, _texts: List[str]):
        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
        vecs = self.client.create_embeddings(self.model, [text])
        return np.array(vecs[0], dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            vecs = self.client.create_embeddings(self.model, chunk)
            out.extend(vecs)
        return np.array(out, dtype=np.float32)

    def embed_batch_with_progress(self, texts: List[str], update: ProgressCb = None) -> np.ndarray:
        total = len(texts)
        out = []
        done = 0
        for i in range(0, total, self.batch):
            chunk = texts[i:i + self.batch]
            vecs = self.client.create_embeddings(self.model, chunk)
            out.extend(vecs)
            done = min(done + len(chunk), total)
            if update:
                update(done, total, "Embedding via engine")
        return np.array(out, dtype=np.float32)