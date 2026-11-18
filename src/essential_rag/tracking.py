"""MLflow tracking helpers: explicit params / metrics / artifacts logging.

Stage 0 relied on ``mlflow.langchain.autolog()`` to log "something" automatically. These helpers
make experiment hygiene *explicit* and intentional: for every evaluation run we record the exact
configuration that produced it (params), the resulting scores (metrics), and the inputs needed to
reproduce or audit it (artifacts). All three functions assume an active run — call them inside a
``with mlflow.start_run(...):`` block.
"""

import mlflow

from .config import Settings


def log_eval_params(settings: Settings, prompt_version: str) -> None:
    """Log the configuration that defines this run.

    Params are the knobs that, together, determine the output: swap any one and you expect
    different metrics. Logging them makes runs comparable and reproducible in the MLflow UI.
    """
    mlflow.log_params(
        {
            "llm_backend": settings.llm_backend,
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "retriever_top_k": settings.retriever_top_k,
            "prompt_version": prompt_version,
        }
    )


def log_eval_artifacts(eval_dataset: list[dict], prompt_template: str) -> None:
    """Log the inputs needed to reproduce or audit the run.

    The eval dataset and the exact prompt text are what you'd need to re-run or explain the
    scores later — so they travel with the run rather than living only in source.
    """
    # log_dict JSON-serializes its first arg; wrap the list so the top level is an object.
    mlflow.log_dict({"examples": eval_dataset}, "eval_dataset.json")
    mlflow.log_text(prompt_template, "prompt.txt")


def log_eval_metrics(metrics: dict) -> None:
    """Log each scorer's aggregate score.

    ``mlflow.genai.evaluate`` returns a flat dict of aggregates (e.g. ``is_not_empty/mean``).
    Only numeric values can be logged as metrics, so non-numeric entries are filtered out.
    """
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if numeric:
        mlflow.log_metrics(numeric)