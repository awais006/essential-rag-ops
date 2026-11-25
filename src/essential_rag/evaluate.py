import logging

import mlflow
import mlflow.langchain
from mlflow.genai import scorer

from . import scorers, tracking
from .chain import PROMPT_VERSION, RAG_PROMPT_TEMPLATE, chain, rag_predict_fn
from .config import get_settings


def main() -> None:
    """Console-script entry point (referenced by pyproject `[project.scripts]`)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    evaluate()


@scorer
def is_not_empty(outputs: str) -> bool:
    """Pass if the model returned a non-empty answer."""
    return scorers.is_not_empty(outputs)


@scorer
def not_hallucinating(outputs: str) -> bool:
    """
    Heuristic: pass if the model does NOT admit it has no context.
    (A proper faithfulness scorer would compare against retrieved docs.)
    """
    return scorers.not_hallucinating(outputs)


@scorer
def keyword_present(inputs: dict, outputs: str, expectations: dict) -> bool:
    """Pass if the expected keyword appears in the output"""
    return scorers.keyword_present(outputs, expectations)


@scorer
def response_length_ok(outputs: str) -> bool:
    """Pass if the response is between 5 and 100 words."""
    return scorers.response_length_ok(outputs)


eval_dataset = [
    {
        "inputs": {"question": "What is MLflow used for?"},
        "expectations": {"keyword": "machine learning"},
    },
    {
        "inputs": {"question": "What does MLflow Tracking log?"},
        "expectations": {"keyword": "metrics"},
    },
    {
        "inputs": {"question": "What is LangChain?"},
        "expectations": {"keyword": "language models"},
    },
    {
        "inputs": {"question": "How does Ollama expose its API?"},
        "expectations": {"keyword": "REST"},
    },
    {"inputs": {"question": "What is RAG?"}, "expectations": {"keyword": "retrieval"}},
]


def evaluate():
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()

    print("Running Essential RAG evaluation ...\n")
    rag_chain = chain()

    with mlflow.start_run(run_name=f"{settings.llm_model}-{PROMPT_VERSION}"):
        # Log config + inputs BEFORE evaluating, so even a crashed run records its context.
        tracking.log_eval_params(settings, PROMPT_VERSION)
        tracking.log_eval_artifacts(eval_dataset, RAG_PROMPT_TEMPLATE)

        results = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=lambda question: rag_predict_fn(question, rag_chain),
            scorers=[is_not_empty, not_hallucinating, keyword_present, response_length_ok],
        )

        tracking.log_eval_metrics(results.metrics)

    print("\n" + "=" * 60)
    print("ESSENTIAL RAG — EVALUATION RESULTS")
    print("=" * 60)
    print("\nAggregate metrics:")
    for k, v in results.metrics.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")
    print("\nPer-sample results:")

    scorer_cols = [c for c in results.result_df.columns if c.endswith("/value")]
    print(results.result_df[["request", "response"] + scorer_cols].to_string(index=False))

    print(settings.mlflow_tracking_uri)
    print("\n OK Done open http://localhost:6001 to view per-sample scores.")

    return results.metrics


def metric_columns(df):
    return sorted(c for c in df.columns if c.startswith("metrics.") and c.endswith("/mean"))


if __name__ == "__main__":
    main()
