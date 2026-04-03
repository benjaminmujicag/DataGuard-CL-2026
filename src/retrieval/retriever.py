"""RAG chain: semantic retrieval over the legal corpus + LLM answer generation.

# Review: opus-4.6 · 2026-04-03
# Reviewed, corrected (error propagation in query_legal), and approved.
"""

import os
from dotenv import load_dotenv

load_dotenv()
for _k, _v in (("ANONYMIZED_TELEMETRY", "False"), ("CHROMA_TELEMETRY", "false")):
    os.environ.setdefault(_k, _v)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from src.ingestion.legal_corpus import format_legal_docs_for_prompt


def get_retriever(top_k: int = 3) -> VectorStoreRetriever:
    """Connect to ChromaDB and return a semantic retriever.

    Args:
        top_k: Number of fragments to retrieve per query.

    Returns:
        A LangChain retriever backed by the local vector store.

    Raises:
        FileNotFoundError: If the ChromaDB persist directory does not exist.
        ConnectionError: If the embedding model cannot be reached via Ollama.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ley_privacidad_cl")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"No se encontró la base de datos en {persist_dir}. "
            "Ejecuta primero: python src/ingestion/ingest.py"
        )

    try:
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
        db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        return db.as_retriever(search_kwargs={"k": top_k})
    except Exception as e:
        raise ConnectionError(
            f"Error al conectar con Ollama ({ollama_base_url}). "
            f"Verifica que Ollama está corriendo con 'ollama serve'. Error: {e}"
        ) from e


def get_rag_chain() -> RunnablePassthrough:  # type: ignore[override]
    """Build the full RAG chain: retriever -> prompt -> LLM -> str output.

    Returns:
        A LangChain RunnableSequence that accepts a question string.

    Raises:
        ConnectionError: If Ollama is unreachable.
    """
    llm_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        llm = OllamaLLM(model=llm_model, base_url=ollama_base_url)
    except Exception as e:
        raise ConnectionError(
            f"No se pudo conectar con Ollama en {ollama_base_url}. "
            f"Verifica que está corriendo con 'ollama serve'. Error: {e}"
        ) from e

    retriever = get_retriever(top_k=int(os.getenv("TOP_K_RESULTS", 3)))

    template = """Eres un experto y riguroso Asesor Legal Corporativo especializado en la normativa chilena, operando estrictamente bajo la Ley 21.719 que Regula la Protección y el Tratamiento de los Datos Personales y crea la Agencia de Protección de Datos Personales de Chile.

Tu tarea es responder a la pregunta del usuario utilizando de manera EXCLUSIVA el conocimiento jurídico proporcionado en "Fragmentos Relevantes".

Reglas de Operación:
1. Responde SIEMPRE en español, de forma clara, consultiva y profesional.
2. Basate ÚNICAMENTE en el contenido de los "Fragmentos Relevantes" para fundamentar tu respuesta.
3. Tienes permitido inferir, deducir y sintetizar conceptos si los fragmentos te dan suficiente contexto tácito (por ejemplo, analizar qué es un dato personal en base a cómo la ley norma su uso).
4. Si los fragmentos de plano no tienen NINGUNA relación con la pregunta o te es imposible deducir una respuesta con ellos, responde cortésmente que la ley no lo menciona explícitamente en el texto recuperado.
5. Cita siempre el Artículo al que haces referencia (y aclara si pertenece a la Ley 19.628 o a la Ley 21.719 repasando la etiqueta 'Fuente' del fragmento).
6. REGLA SUPREMA DE CONFLICTO: Si existe alguna discrepancia, choque de definiciones o actualización entre un fragmento etiquetado como fuente Ley 19.628 y un fragmento de la Ley 21.719, darás SIEMPRE PRIORIDAD ABSOLUTA a la información contenida en la Ley 21.719 por ser la ley modificatoria legalmente vigente.
7. PROHIBIDO inventar normativas externas a los fragmentos o alucinar artículos que no estén en el texto provisto.

---
Fragmentos Relevantes:
{context}

Pregunta del Usuario:
{question}
---

Respuesta Legal:"""

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_legal_docs_for_prompt, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def query_legal(question: str) -> str:
    """High-level function to query the Chilean legal corpus via RAG.

    Args:
        question: Legal question in natural language.

    Returns:
        The LLM-generated answer grounded on retrieved legal fragments.

    Raises:
        ConnectionError: If Ollama or ChromaDB are unreachable.
        FileNotFoundError: If the vector store has not been initialized.
    """
    chain = get_rag_chain()
    return chain.invoke(question)  # type: ignore[return-value]
