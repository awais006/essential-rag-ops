"""FastAPI serving layer for the Essential RAG pipeline.

Exposes the same retrieval+generation chain used in batch evaluation as an online HTTP service.
The chain is built once at startup (lifespan) and reused across requests.
"""
from contextlib import asynccontextmanager
from importlib.metadata import version

import mlflow
import mlflow.langchain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chain import PROMPT_VERSION, chain, rag_answer
from .config import get_settings


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer from the corpus.")


class Source(BaseModel):
    id: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class InfoResponse(BaseModel):
    app_version: str
    llm_backend: str
    llm_model: str
    prompt_version: str
    retriever_top_k: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the chain once at startup; reuse it instead of constructing an LLM client per request.
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()  # each chain.invoke() now emits a trace
    app.state.rag_chain = chain()
    yield
    app.state.rag_chain = None


app = FastAPI(title="Essential RAG API", version=version("essential-rag"), lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Readiness probe: 200 once the app is up and the chain is built, else 503."""
    if getattr(app.state, "rag_chain", None) is None:
        raise HTTPException(status_code=503, detail="chain not initialized")
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a question from the corpus, returning the answer and the source docs used."""
    try:
        answer, docs = rag_answer(req.question, app.state.rag_chain)
    except Exception as exc:  # e.g. Ollama unreachable -> surface as 503, not 500
        raise HTTPException(status_code=503, detail=f"LLM backend error: {exc}") from exc
    return QueryResponse(
        answer=answer,
        sources=[Source(id=d["id"], content=d["content"]) for d in docs],
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Report the active model/prompt configuration for traceability."""
    s = get_settings()
    return InfoResponse(
        app_version=version("essential-rag"),
        llm_backend=s.llm_backend,
        llm_model=s.llm_model,
        prompt_version=PROMPT_VERSION,
        retriever_top_k=s.retriever_top_k,
    )


def serve() -> None:
    """Entry point for the `rag-serve` console script: run the API with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("essential_rag.api:app", host=settings.api_host, port=settings.api_port)











