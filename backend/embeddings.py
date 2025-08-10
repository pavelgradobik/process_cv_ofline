from typing import Optional, Callable, List
from typing import Optional
from scipy.sparse import spmatrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ProgressCb = Optional[Callable[[int, int, str], None]]

class OfflineEmbedder:
    def __init__(self, dimension=384):
        self.vectorizer = TfidfVectorizer(max_features=dimension)
        self.fitted = False
        self.matrix: Optional[spmatrix] = None  # not used directly; kept for lint calm

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() with all texts first!")
        vec_sparse = self.vectorizer.transform([text])
        vec = np.asarray(vec_sparse.todense()).flatten()
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() with all texts first!")
        mat = self.vectorizer.transform(texts)
        return np.asarray(mat.todense())

    def embed_batch_with_progress(self, texts: List[str], update: ProgressCb = None) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() with all texts first!")
        total = len(texts)
        rows = []
        for i, t in enumerate(texts, 1):
            vec_sparse = self.vectorizer.transform([t])
            vec = np.asarray(vec_sparse.todense()).flatten()
            rows.append(vec)
            if update and (i % 200 == 0 or i == total):
                update(i, total, "Vectorizing")
        return np.vstack(rows)

