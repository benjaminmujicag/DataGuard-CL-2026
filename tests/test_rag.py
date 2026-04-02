import os

import pytest
from dotenv import load_dotenv

from src.retrieval.retriever import query_legal


@pytest.mark.integration
def test_rag_chain_questions() -> None:
    """Stress the legal RAG with three representative queries (Ollama + Chroma required)."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    load_dotenv()

    res1 = query_legal("¿Qué es un dato personal según la ley?")
    assert res1 and len(res1) > 20

    res2 = query_legal("¿Cuál es la capital de Francia?")
    assert res2

    res3 = query_legal("¿Qué son los datos sensibles y qué protección especial tienen?")
    assert res3 and len(res3) > 20


if __name__ == "__main__":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    load_dotenv()
    print("--- TEST 1: Elemento legal explícito ---")
    print(query_legal("¿Qué es un dato personal según la ley?"))
    print("--- TEST 2: Elemento fuera de contexto legal ---")
    print(query_legal("¿Cuál es la capital de Francia?"))
    print("--- TEST 3: Sub-tipo de datos explícito ---")
    print(query_legal("¿Qué son los datos sensibles y qué protección especial tienen?"))
