"""Embedding backend -- sentiment.ai is THE embedder, not one option among many.

Concord delegates *all* embedding to its sibling package, sentiment.ai
(`import sentimentai`). That package already owns the model registry (e5 runs
on-device by default; OpenAI is an opt-in paid backend), ships the trained heads,
and carries the provenance/calibration that makes Concord's semantic results
auditable. Concord deliberately does NOT choose its own embedding model -- that
decision lives in sentiment.ai so the two stay in lockstep and a Concord result is
always reproducible against a known sentiment.ai version.

There is no fallback embedder on purpose: a silent swap to a different model would
produce results that look identical but are not comparable or auditable. If
sentiment.ai is absent, semantic features raise a clear install error and the
deterministic lint path (which needs none of this) keeps working.

Importing this module pulls in no ML; sentiment.ai is resolved only when an
embedder is actually requested.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: List[str], kind: str = "passage") -> "object":  # -> np.ndarray (N, D)
        ...


class _SentimentAIEmbedder:
    """Thin adapter over sentimentai.embed_text. Model choice belongs to sentiment.ai.

    e5 models are trained with asymmetric prefixes ("query: " / "passage: "); without
    them retrieval recall drops sharply (measured on this corpus: gdpr 5->8, anonymity
    0->4 @12). We apply the prefix HERE, not in sentiment.ai, so its verified
    sentiment-scoring vectors stay bit-identical. `kind` selects which prefix.
    """

    def __init__(self, model: Optional[str] = None):
        import sentimentai  # sibling package; ImportError handled in get_embedder
        self._embed_text = sentimentai.embed_text
        self._model = model  # None => sentiment.ai's own DEFAULT_MODEL (e5-base)
        self._is_e5 = self._detect_e5(sentimentai, model)

    @staticmethod
    def _detect_e5(sentimentai, model) -> bool:
        if "e5" in (model or "").lower():
            return True
        try:
            b = sentimentai.resolve(model) if model else sentimentai.resolve(sentimentai.DEFAULT_MODEL)
            return "e5" in (getattr(b, "hf_id", "") or "").lower()
        except Exception:
            return model is None  # sentiment.ai's default is an e5 model

    def embed(self, texts: List[str], kind: str = "passage"):
        import numpy as np
        items = list(texts)
        if self._is_e5:
            prefix = "query: " if kind == "query" else "passage: "
            items = [prefix + t for t in items]
        kwargs = {"model": self._model} if self._model else {}
        return np.asarray(self._embed_text(items, **kwargs), dtype="float32")


def get_embedder(model: Optional[str] = None) -> Embedder:
    """Return the sentiment.ai-backed embedder, or raise a clear install error.

    `model` is passed straight through to sentiment.ai's registry (e.g. "e5-small",
    "e5-base", or an OpenAI model); None uses sentiment.ai's default.
    """
    try:
        return _SentimentAIEmbedder(model)
    except ImportError as e:
        raise RuntimeError(
            "Concord's semantic features are powered by the sentiment.ai embedder.\n"
            "  pip install \"concord-ai[embeddings]\"\n"
            f"  import error: {e}"
        ) from e
