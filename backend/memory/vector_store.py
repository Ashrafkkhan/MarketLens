"""
backend/memory/vector_store.py — MarketLens Phase 4
=====================================================
FAISS IndexFlatIP vector store wrapper for historical event embeddings.

Design decisions:
  - IndexFlatIP (inner product) — exact search over L2-normalised vectors
    gives true cosine similarity without approximation.
  - Metadata stored separately in a Parquet file for fast columnar access
    and optional pre-filter before vector search.
  - Index integer IDs map 1-to-1 with Parquet row numbers (0-based).

Persisted files:
  data/faiss_index.bin          — FAISS binary index
  data/faiss_metadata.parquet   — Event metadata (event_id, ticker, AR3, ...)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INDEX_PATH    = DATA_DIR / "faiss_index.bin"
DEFAULT_METADATA_PATH = DATA_DIR / "faiss_metadata.parquet"

EMBEDDING_DIM = 384

# Metadata columns stored alongside the FAISS index
METADATA_COLS = [
    "faiss_id",      # row index in FAISS (int64)
    "event_id",      # PK from events table
    "news_id",       # FK to news table
    "ticker",        # stock ticker or None
    "event_type",    # EARNINGS / MONETARY_POLICY / etc.
    "surprise",      # BEAT / MISS / IN_LINE / NOT_APPLICABLE
    "event_date",    # pandas Timestamp (UTC)
    "summary",       # short text summary
    "formatted_text",# the exact string that was embedded
    "ar_1",          # 1-day abnormal return vs NIFTY 50
    "ar_3",          # 3-day abnormal return vs NIFTY 50  ← primary target
    "ar_5",          # 5-day abnormal return vs NIFTY 50
]


class FAISSVectorStore:
    """
    Thin FAISS wrapper that keeps a metadata DataFrame aligned with the index.

    Thread safety: NOT thread-safe for concurrent writes. Safe for concurrent reads.
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self.index_path    = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dim           = dim
        self._index        = None   # faiss.Index, lazy loaded
        self._metadata: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_new_index(self) -> None:
        """Create a fresh empty IndexFlatIP."""
        import faiss
        self._index = faiss.IndexFlatIP(self.dim)
        logger.info("Initialised new FAISS IndexFlatIP(dim=%d).", self.dim)

    def _require_index(self) -> None:
        """Ensure index and metadata are loaded, raising if not found."""
        if self._index is None or self._metadata is None:
            raise RuntimeError(
                "Vector store is empty or not loaded. "
                "Call build_and_save() or load() first."
            )

    # ------------------------------------------------------------------
    # Build & Persist
    # ------------------------------------------------------------------

    def build_and_save(
        self,
        embeddings: np.ndarray,
        metadata_rows: List[dict],
    ) -> None:
        """
        Build a fresh FAISS index from the provided embeddings and persist
        both the index and metadata to disk.

        Parameters
        ----------
        embeddings : np.ndarray of shape (N, dim), float32, L2-normalised
        metadata_rows : list of dicts — one per embedding row, containing
                        at minimum the keys in METADATA_COLS.
        """
        import faiss

        if len(embeddings) == 0:
            raise ValueError("Cannot build index from zero embeddings.")

        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vecs.shape[1]}")

        # Assign sequential FAISS IDs matching parquet row numbers
        for i, row in enumerate(metadata_rows):
            row["faiss_id"] = i

        self._init_new_index()
        self._index.add(vecs)

        self._metadata = pd.DataFrame(metadata_rows)
        for col in METADATA_COLS:
            if col not in self._metadata.columns:
                self._metadata[col] = None

        # Persist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self._metadata.to_parquet(self.metadata_path, index=False)

        logger.info(
            "Saved FAISS index (%d vectors) → %s",
            self._index.ntotal, self.index_path,
        )
        logger.info("Saved metadata (%d rows) → %s", len(self._metadata), self.metadata_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load persisted index and metadata from disk."""
        import faiss

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run: python scripts/build_vector_index.py"
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata parquet not found at {self.metadata_path}. "
                "Run: python scripts/build_vector_index.py"
            )

        self._index    = faiss.read_index(str(self.index_path))
        self._metadata = pd.read_parquet(self.metadata_path)

        logger.info(
            "Loaded FAISS index: %d vectors | Metadata: %d rows",
            self._index.ntotal, len(self._metadata),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        ticker_filter: Optional[str] = None,
        event_type_filter: Optional[str] = None,
        exclude_event_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve the top-k most similar historical events to the query vector.

        Applies post-search metadata filters (ticker, event_type) and
        self-match exclusion.

        Parameters
        ----------
        query_vector : np.ndarray of shape (1, dim) or (dim,), float32
        top_k : int — number of candidates to retrieve from FAISS
                      (returns up to top_k rows after filters).
        ticker_filter : optional — restrict to events for this ticker
        event_type_filter : optional — restrict to events of this type
        exclude_event_ids : optional list of event_ids to exclude (self-match)

        Returns
        -------
        pd.DataFrame with columns from METADATA_COLS + 'similarity_score',
        sorted descending by similarity_score, up to top_k rows.
        """
        self._require_index()

        vec = np.ascontiguousarray(
            query_vector.reshape(1, self.dim), dtype=np.float32
        )

        # Over-fetch to survive filter pruning
        fetch_k = min(top_k * 4, self._index.ntotal)
        scores, faiss_ids = self._index.search(vec, fetch_k)

        scores   = scores[0]
        faiss_ids = faiss_ids[0]

        # Map FAISS row IDs to metadata rows
        valid_mask = faiss_ids >= 0
        matched = self._metadata.iloc[faiss_ids[valid_mask]].copy()
        matched["similarity_score"] = scores[valid_mask]

        # --- Filters ---
        if exclude_event_ids:
            matched = matched[~matched["event_id"].isin(exclude_event_ids)]

        if ticker_filter:
            matched = matched[matched["ticker"] == ticker_filter]

        if event_type_filter:
            matched = matched[matched["event_type"] == event_type_filter]

        matched = matched.sort_values("similarity_score", ascending=False).head(top_k)
        return matched.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and self._metadata is not None

    @property
    def metadata(self) -> pd.DataFrame:
        self._require_index()
        return self._metadata


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_default_store: Optional[FAISSVectorStore] = None


def get_vector_store(
    index_path: Path = DEFAULT_INDEX_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    auto_load: bool = True,
) -> FAISSVectorStore:
    """
    Return the module-level cached FAISSVectorStore.
    Automatically loads from disk if auto_load=True and files exist.
    """
    global _default_store
    if _default_store is None:
        _default_store = FAISSVectorStore(
            index_path=index_path,
            metadata_path=metadata_path,
        )
    if auto_load and not _default_store.is_loaded:
        if index_path.exists() and metadata_path.exists():
            _default_store.load()
        else:
            logger.warning(
                "Vector store files not found. Call build_vector_index.py first."
            )
    return _default_store
