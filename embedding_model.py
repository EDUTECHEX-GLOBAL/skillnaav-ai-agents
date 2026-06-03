"""
Lazy wrapper around SentenceTransformer.

`sentence_transformers` imports PyTorch which costs ~200 MB of RSS.
Deferring the import until first use keeps the gunicorn worker well
under Render's free-tier 512 MB limit during startup.
"""
from __future__ import annotations
import threading
from typing import Any

_model = None
_util  = None   # sentence_transformers.util — also deferred
_lock  = threading.Lock()  # FIX: prevents concurrent threads from double-loading


def _load():
    global _model, _util
    # Fast path — already loaded (no lock needed once set)
    if _model is not None:
        return
    # Slow path — acquire lock so only ONE thread loads the model
    with _lock:
        if _model is not None:   # re-check inside lock (classic double-checked locking)
            return
        from sentence_transformers import SentenceTransformer, util as _st_util

        # DO NOT pass model_kwargs with low_cpu_mem_usage — it triggers the
        # "Cannot copy out of meta tensor" bug on torch 2.x + transformers 4.3x.
        # The plain constructor is safe and correct.
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        _util  = _st_util


class _LazyEmbedder:
    """
    Drop-in proxy for a SentenceTransformer instance.
    All attribute access and direct calls are forwarded to the real model,
    which is loaded on first use.
    """
    def __getattr__(self, name: str) -> Any:
        _load()
        return getattr(_model, name)

    def __call__(self, *args, **kwargs):
        _load()
        return _model(*args, **kwargs)


class _LazyUtil:
    """Proxy for sentence_transformers.util (cos_sim etc.)."""
    def __getattr__(self, name: str) -> Any:
        _load()
        return getattr(_util, name)


# Public API — import these instead of sentence_transformers directly
embedder = _LazyEmbedder()
util     = _LazyUtil()