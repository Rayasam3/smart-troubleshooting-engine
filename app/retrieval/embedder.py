"""Text -> vector. Two interchangeable backends so the project runs anywhere.

st    : sentence-transformers (semantic: "save my files" ~ "back up data")
tfidf : character n-gram TF-IDF fitted on the catalog (spelling-tolerant, no model download)
"""
import logging

import numpy as np

from app.config import EMBEDDING_BACKEND, EMBEDDING_MODEL

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, backend: str = EMBEDDING_BACKEND, model_name: str = EMBEDDING_MODEL):
        self.backend = backend
        self.model_name = model_name
        self._model = None
        self._tfidf = None
        if backend == "st":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name, device="cpu")
            except Exception as exc:  # no internet / not installed -> degrade, never crash
                log.warning("sentence-transformers unavailable (%s); falling back to tfidf", exc)
                self.backend = "tfidf"

    @property
    def name(self) -> str:
        return f"st:{self.model_name}" if self.backend == "st" else "tfidf:char3-5"

    def fit(self, corpus: list[str]) -> "Embedder":
        """Only the tfidf backend needs fitting (on the catalog texts)."""
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self._tfidf.fit([c.lower() for c in corpus])
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            vecs = self._model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        if self._tfidf is None:
            raise RuntimeError("tfidf backend must be fit() before encode()")
        mat = self._tfidf.transform([t.lower() for t in texts]).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)