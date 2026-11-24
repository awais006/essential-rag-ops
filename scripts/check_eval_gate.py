"""Run the eval with the fake backend (no Ollama / no MLflow server) and fail on low metrics."""

import os
import sys

DEFAULT_THRESHOLD = 0.7


def main() -> int:
    threshold = float(os.getenv("EVAL_GATE_THRESHOLD", DEFAULT_THRESHOLD))
    os.environ.setdefault("LLM_BACKEND", "fake")  # BEFORE settings are read
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow_ci.db")  # local, offline

    from essential_rag.evaluate import evaluate  # import after env is set (lru_cache!)

    means = {k: v for k, v in evaluate().items() if k.endswith("/mean")}

    print(f"\nEval gate threshold: {threshold}")
    failures = [n for n, v in sorted(means.items()) if v < threshold]
    for n, v in sorted(means.items()):
        print(f"  {'PASS' if v >= threshold else 'FAIL'}  {n}: {v:.3f}")
    if failures:
        print(f"\nGate FAILED: {', '.join(failures)} below {threshold}")
        return 1
    print("\nGate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
