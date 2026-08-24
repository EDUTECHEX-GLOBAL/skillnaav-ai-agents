"""
Embedding model — ONNX backend preferred, torch fallback.

On Render (free tier, 512 MB RAM):
  - sentence-transformers >= 3.1.0 + onnxruntime → ~120 MB total  ✓

On local dev (no optimum/onnxruntime installed):
  - Falls back to plain torch SentenceTransformer automatically  ✓
"""
from __future__ import annotations
import threading
import numpy as np
from typing import Any

_model = None
_lock  = threading.Lock()


def _load():
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        from sentence_transformers import SentenceTransformer

        # Try ONNX first (needs sentence-transformers>=3.1.0 + optimum + onnxruntime)
        # Falls back to torch for local dev where optimum isn't installed.
        try:
            _model = SentenceTransformer(
                "paraphrase-MiniLM-L3-v2",
                backend="onnx",
                model_kwargs={"provider": "CPUExecutionProvider"},
            )
            print("[embedder] Loaded with ONNX backend (~120 MB RAM)")
        except (TypeError, Exception):
            # TypeError  → sentence-transformers < 3.1.0 (no backend kwarg)
            # Exception  → optimum/onnxruntime not installed
            _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            print("[embedder] Loaded with torch backend (ONNX deps not available)")


class _LazyEmbedder:
    def __getattr__(self, name: str) -> Any:
        _load()
        return getattr(_model, name)

    def __call__(self, *args, **kwargs):
        _load()
        kwargs.pop("convert_to_tensor", None)
        return _model(*args, **kwargs)


class _CosSim:
    """Numpy cos_sim — works with both numpy arrays (ONNX) and torch tensors."""
    def __call__(self, a, b):
        def to_np(x):
            if hasattr(x, "numpy"):
                x = x.numpy()
            return np.atleast_2d(np.array(x, dtype=np.float32))
        a, b = to_np(a), to_np(b)
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        return a @ b.T

    def __getattr__(self, name: str) -> Any:
        from sentence_transformers import util as _st_util
        return getattr(_st_util, name)


class _LazyUtil:
    _cos = _CosSim()

    @property
    def cos_sim(self):
        return self._cos

    def __getattr__(self, name: str) -> Any:
        _load()
        from sentence_transformers import util as _st_util
        return getattr(_st_util, name)


embedder = _LazyEmbedder()
util     = _LazyUtil()