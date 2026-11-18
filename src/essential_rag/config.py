"""Centralized, environment-driven configuration.

All runtime settings live here instead of being hardcoded across modules. Values are read from
environment variables (or a local ``.env`` file), each with a sensible default so the project runs
out of the box with local Ollama. Secrets are wrapped in ``SecretStr`` so they are not accidentally
printed or logged.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment / a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(default="http://localhost:6001")
    mlflow_experiment: str = Field(default="essential-rag")

    # --- LLM backend selection ---
    llm_backend: Literal["ollama", "openai"] = Field(default="ollama")
    llm_model: str = Field(default="qwen3.5:0.8b")
    llm_temperature: float = Field(default=0.1)

    # --- Ollama backend (used when llm_backend == "ollama") ---
    ollama_base_url: str = Field(default="http://localhost:11434")

    # --- OpenAI-compatible backend (used when llm_backend == "openai") ---
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_api_key: SecretStr = Field(default=SecretStr(""))

    # --- Retriever ---
    retriever_top_k: int = Field(default=2)

    # --- API serving ---
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance so every module shares one source of truth.

    The cache means the environment is read once per process. Tests that need to override env
    vars can call ``get_settings.cache_clear()`` after patching the environment.
    """
    return Settings()