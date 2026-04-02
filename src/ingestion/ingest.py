import os
import argparse
from dotenv import load_dotenv

# Carga temprana para que ChromaDB respete su inhabilitación de telemetría ANTES de importarlo
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def load_pdfs(pdf_dir: str) -> list[Document]:
    """Carga todos los PDFs de un directorio y extrae su texto inyectando metadata.

    Args:
        pdf_dir: Ruta al directorio que contiene los archivos PDF (Leyes).

    Returns:
        Lista de documentos LangChain listos para su uso.
    """
    if not os.path.exists(pdf_dir) or not os.listdir(pdf_dir):
        raise FileNotFoundError(f"No se encontraron archivos en el directorio: {pdf_dir}")
        
    print(f"Cargando biblioteca de leyes desde: {pdf_dir}...")
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    print(f"✅ Se cargaron bibliotecas base: {len(documents)} páginas indexadas desde los archivos.")
    return documents

def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Divide los documentos en fragmentos (chunks) más manejables.

    Args:
        documents: Lista de documentos a dividir.
        chunk_size: Tamaño máximo de cada chunk en caracteres.
        chunk_overlap: Cantidad de caracteres superpuestos entre chunks consecutivos.

    Returns:
        Lista de fragmentos de documentos listos para su vectorización.
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

def store_in_chroma(chunks: list[Document], collection_name: str, persist_dir: str, embedding_model: str, base_url: str) -> Chroma:
    """Genera embeddings vectoriales de los chunks y los persiste en ChromaDB.

    Args:
        chunks: Lista de fragmentos de documentos vectorizables.
        collection_name: Nombre de la base de datos vectorial a crear o cargar.
        persist_dir: Ruta del directorio local para persistencia física de la base de datos.
        embedding_model: Modelo de generación de embeddings (ex: nomic-embed-text).
        base_url: URL base local de Ollama.

    Returns:
        Instancia de la base de datos Chroma con los embeddings y documentos cargados.
    """
    print("Iniciando conexión con Ollama para almacenar en ChromaDB...")
    try:
        embeddings = OllamaEmbeddings(
            model=embedding_model, 
            base_url=base_url
        )
    except Exception as e:
        raise ConnectionError(
            f"No se pudo conectar con Ollama en {base_url}. "
            f"Verifica que Ollama está corriendo con 'ollama serve'. Error: {e}"
        )

    print(f"Generando arreglos vectoriales en coleccion '{collection_name}' (esto tomará algunos minutos)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    print(f"✅ Embeddings almacenados en ChromaDB ({persist_dir}/)")
    return vector_store

def ingest(force: bool = False) -> None:
    """Función principal para orquestar la ingesta del PDF a ChromaDB. Idempotente.
    
    Args:
         force: Si es True, ignora validación de existencia e inyecta los embeddings de manera forzada.
    """
    # load_dotenv() ya fue invocado arriba del archivo
    
    pdf_dir = os.getenv("PDF_DIR", "./data/leyes_base")
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ley_privacidad_cl")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Verificación de idempotencia
    if not force and os.path.exists(persist_dir):
        try:
             # Comprobar si ya hay embeddings instanciando db nativamente
             embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
             db = Chroma(collection_name=collection_name, persist_directory=persist_dir, embedding_function=embeddings)
             doc_count = len(db.get()["ids"]) # type: ignore
             if doc_count > 0:
                 print(f"ℹ️ La colección '{collection_name}' ya contiene {doc_count} documentos. No se requiere re-ingesta.")
                 print("(Usa la bandera --force para ignorar e ingestar forzosamente)")
                 return
        except Exception as e:
             print(f"Advertencia al chequear idempotencia de ChromaDB: {e}. Procesando ingesta normalmente por precaución.")

    docs = load_pdfs(pdf_dir)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    db = store_in_chroma(chunks, collection_name, persist_dir, embedding_model, ollama_base_url)
    
    # Comprobación de integridad para mensaje final en la terminal
    doc_count = len(db.get()["ids"]) # type: ignore
    print(f"✅ Colección '{collection_name}' operando correctamente con {doc_count} documentos en total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de ingesta de la Base Legal hacia ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Ignora verificación de volumen pre-existente y re-ingesta el PDF a la base vectorial")
    args = parser.parse_args()
    
    ingest(force=args.force)
