"""Health check del entorno: verifica comunicación con Ollama (LLM + Embeddings).

Ejecutar desde la raíz del proyecto:
    python tests/test_setup.py
"""
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Cargar variables de entorno desde .env (con LangSmith desactivado si no hay key)
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
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector = embeddings.embed_query("prueba")
    print(f"✅ Embeddings funcionan: vector de {len(vector)} dimensiones")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Verifica que Ollama está corriendo con: ollama serve")
