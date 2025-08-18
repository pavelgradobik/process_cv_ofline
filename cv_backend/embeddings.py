from typing import List, Optional
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from cv_backend.config import TFIDF_MAX_FEATURES

class OfflineEmbedder:
    """
    TF-IDF baseline with stronger defaults:
      - unigrams + bigrams
      - english stopwords
      - sublinear tf
      - token pattern keeps slashes/hyphens (ui/ux, server-side)
    """
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\-\/]+\b",
            max_df=0.95,
            min_df=2,
        )
        self._fitted = False
        self._matrix: Optional[spmatrix] = None

    def fit(self, corpus: List[str]) -> None:
        self._matrix = self.vectorizer.fit_transform(corpus)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OfflineEmbedder not fitted. Call fit(corpus) first.")
        mat = self.vectorizer.transform(texts)
        return mat.toarray().astype(np.float32)

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)