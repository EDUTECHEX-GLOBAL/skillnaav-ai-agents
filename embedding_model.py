from sentence_transformers import SentenceTransformer

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

# Keep `embedder` as a lazy proxy so existing `from embedding_model import embedder`
# calls still work without changes elsewhere.
class _LazyEmbedder:
    def __getattr__(self, name):
        return getattr(_get_embedder(), name)
    def __call__(self, *args, **kwargs):
        return _get_embedder()(*args, **kwargs)

embedder = _LazyEmbedder()