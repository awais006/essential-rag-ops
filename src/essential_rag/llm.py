"""Pluggable LLM backend factory.

``get_llm`` builds a LangChain language model from the active :class:`Settings`. Local Ollama is the
default so the project runs with no cloud account; set ``LLM_BACKEND=openai`` to point at any
OpenAI-compatible endpoint instead. Provider classes are imported lazily inside each branch so you
only need the package for the backend you actually use.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> BaseLanguageModel:
    """Build the language model selected by ``settings.llm_backend``.

    Args:
        settings: Settings to use. Falls back to the shared cached settings when not provided.

    Returns:
        A LangChain language model ready to compose into a chain (``prompt | llm``).

    Raises:
        ValueError: If ``settings.llm_backend`` is not a supported backend.
    """
    settings = settings or get_settings()

    if settings.llm_backend == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    if settings.llm_backend == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
        )

    if settings.llm_backend == "fake":
        return FakeLLM()

    raise ValueError(
        f"Unsupported LLM_BACKEND: {settings.llm_backend!r}. Expected 'ollama' or 'openai'."
    )


class FakeLLM(LLM):
    """Deterministic, offline LLM for CI: answers from the prompt's Context block."""

    max_words: int = 90

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        context = self._between(prompt, "Context:", "Question:")
        words = context.split()
        if len(words) < 5:
            return "The provided context does not give enough detail to answer precisely."
        return " ".join(words[: self.max_words])

    @staticmethod
    def _between(text: str, start: str, end: str) -> str:
        return text.split(start, 1)[-1].split(end, 1)[0].strip()
