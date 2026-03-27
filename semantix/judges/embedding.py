"""Fast embedding-based judge using sentence-transformers cosine similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from semantix.judges import Judge, Verdict

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingJudge(Judge):
    """Judge that uses cosine similarity between sentence embeddings.

    This is the *fast* validation path — no LLM round-trip required.

    Parameters
    ----------
    model_name:
        Any model identifier accepted by ``sentence_transformers.SentenceTransformer``.
        Defaults to ``"all-MiniLM-L6-v2"`` (fast, ~80 MB).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Lazy import so that sentence-transformers is only required when
        # this judge is actually used.
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self._model: SentenceTransformer = SentenceTransformer(model_name)

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        from sentence_transformers.util import cos_sim  # noqa: WPS433

        embeddings = self._model.encode(
            [output, intent_description],
            convert_to_tensor=True,
        )
        score: float = cos_sim(embeddings[0], embeddings[1]).item()
        return Verdict(
            passed=score >= threshold,
            score=score,
        )
