from essential_rag.retriever import retrieve


def test_returns_top_k_count():
    # default k = 2
    assert len(retrieve("What is MLflow?")) == 2
    # explicit k is respected
    assert len(retrieve("What is MLflow?", top_k=1)) == 1
    assert len(retrieve("What is MLflow?", top_k=3)) == 3


def test_ranks_relevant_doc_first():
    # an MLflow question should surface an MLflow doc at the top
    top = retrieve("What is MLflow used for?", top_k=1)[0]
    assert "mlflow" in top["id"]

    # a LangChain question should surface a LangChain doc
    top = retrieve("LangChain chains agents memory abstractions", top_k=1)[0]
    assert "langchain" in top["id"]


def test_return_shape():
    docs = retrieve("Ollama REST API", top_k=2)
    assert isinstance(docs, list)
    assert all("id" in d and "content" in d for d in docs)


def test_is_deterministic():
    q = "What does MLflow Tracking log?"
    assert retrieve(q) == retrieve(q)  # same input -> same output
