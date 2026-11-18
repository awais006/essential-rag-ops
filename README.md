# Essential RAG

A minimal, dependency-light **Retrieval-Augmented Generation (RAG)** pipeline with **automated LLM evaluation**. It wires together a tiny keyword retriever, a pluggable LLM backend (local **Ollama** or any **OpenAI-compatible** endpoint), and a **LangChain** generation chain, then scores the answers with **MLflow GenAI** evaluation.

It is built to be read end-to-end: no vector database, no cloud account required, and every moving part lives in one small module.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Full RAG loop** — retrieve relevant context, then generate a grounded answer.
- **Pluggable LLM backend** — local Ollama by default; switch to OpenAI (or any compatible endpoint) with one env var.
- **Automated evaluation** — runs a question set through the pipeline and scores every answer with four heuristic scorers via `mlflow.genai.evaluate`.
- **Environment-driven config** — all settings come from env vars / a `.env` file (Pydantic `BaseSettings`), nothing hardcoded.
- **Dependency-free retrieval** — keyword-overlap retriever over an in-memory corpus, so the demo runs anywhere.
- **Tested** — pure scorer and retriever logic covered by `pytest`.

---

## Architecture

The full workflow — runtime flow, component reference, and an end-to-end walkthrough — is in [`workflow.svg`](./workflow.svg):

![Code workflow diagram](./workflow.svg)

**Data flow in one line:**

```
question → retrieve(CORPUS) → top-2 docs → context → prompt | llm | parser → answer → 4 scorers → MLflow metrics
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Config | `config.py` | `Settings` / `get_settings()` — load settings from env/`.env`, cached once per process. |
| LLM factory | `llm.py` | `get_llm()` — builds `OllamaLLM` or `ChatOpenAI` from settings. |
| Knowledge base | `corpus.py` | `CORPUS` — 7 in-memory docs `{id, content}`. |
| Retrieval | `retriever.py` | `retrieve(query, top_k=2)` — rank corpus by keyword overlap. |
| Generation | `chain.py` | `chain()` builds `prompt \| llm \| StrOutputParser`; `rag_predict_fn()` runs retrieve → generate. |
| Scoring | `scorers.py` | Pure pass/fail metric functions. |
| Orchestration | `evaluate.py` | Entry point; wires the chain, dataset, and scorers into `mlflow.genai.evaluate`. |

---

## Project structure

```
Essential RAG/
├── src/essential_rag/
│   ├── config.py        # Settings + get_settings() (env-driven)
│   ├── llm.py           # get_llm() — Ollama / OpenAI factory
│   ├── corpus.py        # CORPUS: in-memory knowledge base
│   ├── retriever.py     # retrieve() — keyword-overlap retriever
│   ├── chain.py         # chain() + rag_predict_fn() — the RAG pipeline
│   ├── scorers.py       # pure scorer logic
│   └── evaluate.py      # main() entry point + MLflow evaluation
├── tests/
│   ├── test_retriever.py
│   └── test_scorers.py
├── workflow.svg         # architecture + walkthrough diagram
├── .env.example         # configuration template
├── pyproject.toml
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`
- **An MLflow tracking server** (started locally below)
- **A model backend**, either:
  - **[Ollama](https://ollama.com/)** running locally (default), or
  - an **OpenAI-compatible API** key.

---

## Installation

```bash
# Clone
git clone <your-repo-url>
cd "Essential RAG"

# Install with uv (creates the virtualenv and installs deps + dev tools)
uv sync
```

<details>
<summary>Using pip instead of uv</summary>

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```
</details>

---

## Configuration

Copy the template and edit as needed:

```bash
cp .env.example .env
```

Key settings (all optional — sensible defaults ship for local Ollama):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `http://localhost:6001` | MLflow tracking server. |
| `MLFLOW_EXPERIMENT` | `essential-rag` | Experiment name in the MLflow UI. |
| `LLM_BACKEND` | `ollama` | `ollama` or `openai`. |
| `LLM_MODEL` | `qwen3.5:0.8b` | Model tag/name passed to the backend. |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (when `LLM_BACKEND=ollama`). |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base (when `LLM_BACKEND=openai`). |
| `OPENAI_API_KEY` | _(empty)_ | API key (when `LLM_BACKEND=openai`). Keep it secret. |

> `.env` is git-ignored — never commit real keys.

---

## Running the evaluation

**1. Start an MLflow tracking server** (in its own terminal):

```bash
mlflow server --port 6001
```

**2. Prepare the model backend.**

For the default Ollama backend, pull the model:

```bash
ollama pull qwen3.5:0.8b
```

For the OpenAI backend instead, set `LLM_BACKEND=openai` and `OPENAI_API_KEY` in `.env`.

**3. Run the pipeline evaluation:**

```bash
# via the installed console script
uv run essential-rag

# or as a module
uv run python -m essential_rag.evaluate
```

You'll get aggregate metrics and a per-sample table in the console, and a full run logged to the MLflow UI at **http://localhost:6001**.

---

## How it works

1. **Entry** — the `essential-rag` console script calls `evaluate.main()`, which configures logging and calls `evaluate()`.
2. **Setup** — `evaluate()` points MLflow at the tracking server, sets the experiment, and enables `mlflow.langchain.autolog()`.
3. **Build the chain once** — `rag_chain = chain()` composes `prompt | llm | StrOutputParser()`, where `llm` comes from `get_llm()` reading the active `Settings`.
4. **Evaluate** — `mlflow.genai.evaluate` runs each question in `eval_dataset` through `rag_predict_fn(question, rag_chain)`:
   - `retrieve(question)` ranks the corpus by keyword overlap and returns the top 2 docs,
   - their content is joined into a `context` string,
   - `rag_chain.invoke({question, context})` fills the prompt, the LLM generates, and the parser returns a plain-text answer.
5. **Score & report** — four scorers grade each answer; MLflow aggregates the metrics and per-sample results, which are printed and logged to the UI.

### Scorers

| Scorer | Passes when… |
|--------|--------------|
| `is_not_empty` | the answer is non-empty / not whitespace. |
| `not_hallucinating` | the answer does **not** contain a refusal phrase (e.g. "I don't know based on…"). |
| `keyword_present` | the expected keyword for that question appears in the answer. |
| `response_length_ok` | the answer is between **5 and 100** words. |

---

## Testing

The retriever and scorers are pure functions, so they test fast with no LLM or server:

```bash
uv run pytest
```

---

## Switching the LLM backend

The backend is selected entirely by configuration — no code changes:

```dotenv
# .env — use an OpenAI-compatible endpoint
LLM_BACKEND=openai
LLM_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
```

`get_llm()` reads these and builds the matching client; the rest of the pipeline is unchanged.

---

## License

[MIT](./LICENSE) © Muhammad Awais
