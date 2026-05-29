"""
Lazy wrapper around SentenceTransformer.

`sentence_transformers` imports PyTorch which costs ~200 MB of RSS.
Deferring the import until first use keeps the gunicorn worker well
under Render's free-tier 512 MB limit during startup.
changes
"""
from __future__ import annotations
from typing import Any

_model = None
_util  = None   # sentence_transformers.util — also deferred


def _load():
    global _model, _util
    if _model is None:
        from sentence_transformers import SentenceTransformer, models, util as _st_util
        
        # In sentence-transformers 2.6.1, we must pass low_cpu_mem_usage via models.Transformer
        word_embedding_model = models.Transformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2", 
            model_args={"low_cpu_mem_usage": False}
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        _model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
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