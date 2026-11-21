"""Pure scorer logic — no framework dependency, so it's fast and trivial to unit-test."""


def is_not_empty(outputs: str) -> bool:
    """Pass if the model returned a non-empty answer."""
    return bool(outputs and outputs.strip())


def not_hallucinating(outputs: str) -> bool:
    """Pass if the model does NOT admit it has no context."""
    refusal_phrases = ["i don't know based on", "not in the context", "cannot find"]
    return not any(p in outputs.lower() for p in refusal_phrases)


def keyword_present(outputs: str, expectations: dict) -> bool:
    """Pass if the expected keyword appears in the output."""
    kw = expectations.get("keyword", "")
    return kw.lower() in outputs.lower() if kw else False


def response_length_ok(outputs: str) -> bool:
    """Pass if the response is between 5 and 100 words."""
    wc = len(outputs.split())
    return 5 <= wc <= 100
