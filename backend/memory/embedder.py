"""
backend/memory/embedder.py — MarketLens Phase 4
================================================
Local sentence-transformer embedding service for financial event text.

Encodes combined structured + semantic event strings into dense L2-normalised
vectors suitable for inner-product (cosine) similarity search via FAISS.

Embedding format:
    "[{ticker}] {event_type} | Surprise: {surprise} | {summary}"

Model: sentence-transformers/all-MiniLM-L6-v2  (384-dim, ~90 MB, CPU-friendly)
Output dimension: 384
Normalisation: L2 (unit vectors → inner product = cosine similarity)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64


def _format_event_text(
    ticker: Optional[str],
    event_type: str,
    surprise: Optional[str],
    summary: str,
) -> str:
    """
    Build a structured embedding string that fuses categorical context
    with free-text semantics for richer vector representations.

    Example:
        "[INFY] EARNINGS | Surprise: BEAT | Infosys Q2 profit jumps 12%..."
    """
    ticker_str = ticker if ticker else "MACRO"
    surprise_str = surprise if surprise else "NOT_APPLICABLE"
    summary_str = (summary or "").strip()
    return f"[{ticker_str}] {event_type} | Surprise: {surprise_str} | {summary_str}"


class EventEmbedder:
    """
    Wraps a local SentenceTransformer model to produce L2-normalised embeddings
    for financial event strings, ready for FAISS IndexFlatIP cosine search.

    Lazy-loads the model on first call to avoid import-time GPU/CPU init.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None  # Lazy init

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Loads the SentenceTransformer model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info("Loading embedding model '%s'...", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded. Output dimension: %d", EMBEDDING_DIM)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of raw strings into a float32 matrix of shape (N, dim).

        Parameters
        ----------
        texts : List[str]
            Raw strings to embed. Use _format_event_text() for event strings.
        normalize : bool
            If True (default), L2-normalise each vector so that
            inner product == cosine similarity.
        show_progress : bool
            Show tqdm progress bar during batch encoding.

        Returns
        -------
        np.ndarray of shape (len(texts), 384), dtype float32
        """
        self._load_model()

        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        logger.debug("Encoding %d texts in batches of %d...", len(texts), self.batch_size)

        vecs: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,  # SentenceTransformer handles L2 norm
        ).astype(np.float32)

        # Safety: re-normalise in case model version doesn't support parameter
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms

        logger.debug("Encoded shape: %s", vecs.shape)
        return vecs

    def embed_events(
        self,
        records: List[dict],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, List[str]]:
        """
        Convenience wrapper: builds formatted embedding strings from event
        record dicts and returns (embeddings, formatted_texts).

        Each record dict must contain:
            - ticker      (str or None)
            - event_type  (str)
            - surprise    (str or None)
            - summary     (str)

        Returns
        -------
        (embeddings: np.ndarray[N, 384], formatted_texts: List[str])
        """
        texts = [
            _format_event_text(
                r.get("ticker"),
                r.get("event_type", "OTHER"),
                r.get("surprise"),
                r.get("summary", ""),
            )
            for r in records
        ]
        embeddings = self.embed_texts(texts, normalize=normalize, show_progress=show_progress)
        return embeddings, texts

    def embed_query(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        surprise: Optional[str] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a single query headline/text as a 1-row array (1, 384).
        Optionally enriches query with structured context fields.

        If event_type and surprise are provided, the query is formatted
        identically to historical event strings for fair comparison.
        """
        if event_type:
            text = _format_event_text(ticker, event_type, surprise, query_text)
        else:
            text = query_text

        return self.embed_texts([text], normalize=normalize)


# ---------------------------------------------------------------------------
# Module-level singleton (shared across the process)
# ---------------------------------------------------------------------------
_default_embedder: Optional[EventEmbedder] = None


def get_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> EventEmbedder:
    """Return the module-level cached EventEmbedder (lazy-init)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = EventEmbedder(model_name=model_name)
    return _default_embedder
