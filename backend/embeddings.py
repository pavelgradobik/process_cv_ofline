from typing import Optional
from scipy.sparse import spmatrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class OfflineEmbedder:
    def __init__(self, dimension=384):
        self.vectorizer = TfidfVectorizer(max_features=dimension)
        self.fitted = False
        self.matrix: Optional[spmatrix] = None

    def fit(self, texts):
        self.matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True

    def embed(self, text):
        if not self.fitted or self.matrix is None:
            raise RuntimeError("Call fit() with all texts first!")
        vec_sparse = self.vectorizer.transform([text])
        vec = np.asarray(vec_sparse.todense()).flatten()
        if vec.shape[0] < self.matrix.shape[1]:
            pad = np.zeros(self.matrix.shape[1] - vec.shape[0])
            vec = np.concatenate([vec, pad])
        return vec

    def embed_batch(self, texts):
        return np.vstack([self.embed(t) for t in texts])

