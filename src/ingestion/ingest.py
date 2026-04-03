"""Ingest the legal Markdown corpus into ChromaDB for RAG retrieval.

Reads cleaned ``.md`` files from ``LEGAL_CORPUS_DIR`` (default ``./data/leyes_base``),
splits them into chunks and stores embeddings in a local ChromaDB collection.
The script is **idempotent**: it skips ingestion when the collection already has data
unless ``--force`` is passed.
"""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()
for _k, _v in (("ANONYMIZED_TELEMETRY", "False"), ("CHROMA_TELEMETRY", "false")):
    os.environ.setdefault(_k, _v)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from src.ingestion.legal_corpus import load_legal_corpus, resolve_legal_md_paths


def load_legal_docs(corpus_dir: str) -> list[Document]:
    """Load the fixed legal corpus (3 Markdown files) with 21.719 > mixta > 19.628 priority.

    Args:
        corpus_dir: Directory that must contain ``ley_21719.md``, ``ley_mixta_sucia.md``
            and ``ley_19628.md`` (or accepted alias). See ``legal_corpus.py``.

    Returns:
        List of LangChain Documents ready for chunking.
    """
    resolved = resolve_legal_md_paths(corpus_dir)
    print(f"Cargando corpus legal desde: {corpus_dir}...")
    for path, label, _ in resolved:
        print(f"  · {path.name} ({label})")
    documents = load_legal_corpus(corpus_dir)
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"✅ Se cargaron {len(documents)} documentos ({total_chars:,} caracteres totales).")
    return documents


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of documents to split.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of document fragments ready for vectorization.
    """
    print(f"Dividiendo textos en chunks de {chunk_size} (con {chunk_overlap} de overlap)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Chunks generados: {len(chunks)} fragmentos")
    return chunks


def store_in_chroma(
    chunks: list[Document],
    collection_name: str,
    persist_dir: str,
    embedding_model: str,
    base_url: str,
) -> Chroma:
    """Generate embeddings and persist them in ChromaDB.

    Args:
        chunks: Document fragments to vectorize.
        collection_name: Name of the ChromaDB collection.
        persist_dir: Local directory for ChromaDB persistence.
        embedding_model: Ollama embedding model name.
        base_url: Ollama base URL.

    Returns:
        ChromaDB vector store instance with the loaded embeddings.
    """
    print("Iniciando conexión con Ollama para almacenar en ChromaDB...")
    try:
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    except Exception as e:
        raise ConnectionError(
            f"No se pudo conectar con Ollama en {base_url}. "
            f"Verifica que Ollama está corriendo con 'ollama serve'. Error: {e}"
        )

    print(f"Generando embeddings en colección '{collection_name}' (puede tomar varios minutos)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    print(f"✅ Embeddings almacenados en ChromaDB ({persist_dir}/)")
    return vector_store


def ingest(force: bool = False) -> None:
    """Orchestrate the full ingestion pipeline. Idempotent unless ``force=True``.

    Args:
        force: If True, skip the existing-data check and re-ingest from scratch.
    """
    corpus_dir = os.getenv("LEGAL_CORPUS_DIR", os.getenv("PDF_DIR", "./data/leyes_base"))
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ley_privacidad_cl")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if not force and os.path.exists(persist_dir):
        try:
            embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            db = Chroma(
                collection_name=collection_name,
                persist_directory=persist_dir,
                embedding_function=embeddings,
            )
            doc_count = len(db.get()["ids"])  # type: ignore[index]
            if doc_count > 0:
                print(
                    f"ℹ️ La colección '{collection_name}' ya contiene {doc_count} documentos. "
                    "No se requiere re-ingesta."
                )
                print("(Usa la bandera --force para ignorar e ingestar forzosamente)")
                return
        except Exception as e:
            print(
                f"Advertencia al chequear idempotencia de ChromaDB: {e}. "
                "Procesando ingesta normalmente por precaución."
            )

    docs = load_legal_docs(corpus_dir)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    db = store_in_chroma(chunks, collection_name, persist_dir, embedding_model, ollama_base_url)

    doc_count = len(db.get()["ids"])  # type: ignore[index]
    print(f"✅ Colección '{collection_name}' operando correctamente con {doc_count} documentos en total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta del corpus legal (.md) hacia ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora verificación de volumen pre-existente y re-ingesta a la base vectorial",
    )
    args = parser.parse_args()
    ingest(force=args.force)
