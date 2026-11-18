from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from .llm import get_llm
from .retriever import retrieve

# Bump PROMPT_VERSION whenever RAG_PROMPT_TEMPLATE changes so runs stay comparable in MLflow.
PROMPT_VERSION = "v1"

RAG_PROMPT_TEMPLATE = (
    "Use the following context to answer the question. "
    "If the context does not contain the answer, say 'I don't know based on the context.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def chain() -> Runnable:
    llm = get_llm()

    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    return rag_prompt | llm | StrOutputParser()


def rag_answer(question: str, rag_chain: Runnable) -> tuple[str, list[dict]]:
    """Retrieve context and generate an answer; return both the answer and the source docs."""
    docs = retrieve(question)
    context = "\n\n".join(d["content"] for d in docs)
    answer = rag_chain.invoke({"question": question, "context": context})
    return answer, docs


def rag_predict_fn(question: str, rag_chain: Runnable) -> str:
    """Full RAG pipeline: retrieve -> generate (answer only, for batch eval)."""
    answer, _ = rag_answer(question, rag_chain)
    return answer
