"""Health check del entorno: verifica comunicación con Ollama (LLM + Embeddings).

Ejecutar desde la raíz del proyecto:
    python tests/test_setup.py
"""

import os

import pytest
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM


@pytest.mark.integration
def test_ollama_llm_and_embeddings() -> None:
    """Valida LLM y embeddings locales (requiere Ollama en marcha)."""
    load_dotenv()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = OllamaLLM(model=ollama_model, base_url=base_url)
    response = llm.invoke("Responde solo con 'OK' si puedes leerme.")
    assert response
    assert "OK" in str(response).upper() or len(str(response)) > 0

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    vector = embeddings.embed_query("prueba")
    assert len(vector) >= 64


if __name__ == "__main__":
    load_dotenv()
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        print("Testing LLM (puede tardar unos segundos en cargar el modelo en memoria)...")
        llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        response = llm.invoke("Responde solo con 'OK' si puedes leerme.")
        print(f"✅ LLM responde: {response[:50]}")

        print("Testing Embeddings...")
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )
        vector = embeddings.embed_query("prueba")
        print(f"✅ Embeddings funcionan: vector de {len(vector)} dimensiones")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Verifica que Ollama está corriendo con: ollama serve")
