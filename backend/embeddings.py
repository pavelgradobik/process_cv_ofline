from typing import Optional, Callable, List
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.config import EMBEDDING_MODEL, EMBED_BATCH
from backend.engine_client import EngineClient

ProgressCb = Optional[Callable[[int, int, str], None]]

class OfflineEmbedder:
    """
    Stronger TF-IDF baseline:
      - sublinear tf
      - unigrams + bigrams
      - english stopwords
      - keep slashes/hyphens to preserve 'ui/ux', 'python-dev'
    """
    def __init__(self, max_features: int = 50000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\-\/]+\b",
            max_df=0.95,
            min_df=2
        )
        self.fitted = False
        self.matrix: Optional[spmatrix] = None

    def fit(self, documents: List[str]) -> None:
        self.matrix = self.vectorizer.fit_transform(documents)
        self.fitted = True

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OfflineEmbedder not fitted. Call fit(texts) first.")
        mat = self.vectorizer.transform(texts)
        return mat.toarray().astype(np.float32)

class EngineEmbedder:

    def __init__(self, model: str = EMBEDDING_MODEL, batch: int = EMBED_BATCH):
        self.client = EngineClient()
        self.model = model
        self.batch = batch

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
