"""
backend/intelligence/analysis_cache.py — persistent cache for /analyze results.

Stores finished analyses in a JSON file so a pre-analysed headline is served
instantly after a server restart (used for demos). The key is derived from the
headline, body and ticker override.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "demo_cache" / "analysis_cache.json"
_lock = threading.Lock()


def make_key(headline: str, body: Optional[str] = "", ticker: Optional[str] = None) -> str:
    raw = "|".join([
        " ".join((headline or "").lower().split()),
        " ".join((body or "").lower().split()),
        (ticker or "").split(".")[0].strip().upper(),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _read() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Analysis cache unreadable (%s); ignoring.", e)
        return {}


def get_cached(key: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _read().get(key)


def save_cached(key: str, result: Dict[str, Any]) -> None:
    with _lock:
        data = _read()
        data[key] = result
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(CACHE_PATH)
