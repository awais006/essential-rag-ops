from .config import get_settings
from .corpus import CORPUS


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Return the top_k corpus entries with the most keyword overlaps.

    When ``top_k`` is None, fall back to the configured ``retriever_top_k`` so the value
    has one source of truth (and can be logged as an MLflow param).
    """
    if top_k is None:
        top_k = get_settings().retriever_top_k

    query_words = set(query.lower().split())
    scored = []
    for doc in CORPUS:
        doc_words = set(doc["content"].lower().split())
        overlap = len(query_words & doc_words)
        scored.append((overlap, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
