import os
from dotenv import load_dotenv

# Cargar variables de entorno ANTES de importar ChromaDB para que lea CHROMA_TELEMETRY=false
load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever(top_k: int = 3) -> VectorStoreRetriever:
    """Conecta a ChromaDB y devuelve un retriever para la búsqueda semántica.

    Args:
        top_k: Cantidad de fragmentos recuperados a retornar.

    Returns:
        Un objeto retriever de LangChain configurado para usar base de datos vectorial local.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ley_privacidad_cl")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Validar que exista la base de datos
    if not os.path.exists(persist_dir):
         raise FileNotFoundError(
             f"No se encontró la base de datos en {persist_dir}. "
             "Asegúrate de haber ejecutado el script de ingesta (F1.2) primero."
         )

    try:
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
        db = Chroma(
            collection_name=collection_name, 
            persist_directory=persist_dir, 
            embedding_function=embeddings
        )
        return db.as_retriever(search_kwargs={"k": top_k})
    except Exception as e:
        raise ConnectionError(
             f"Error al conectar con el motor local de embeddings a través de Ollama. Error detallado: {e}"
        )

def get_rag_chain() -> RunnablePassthrough: # type: ignore
    """Construye la cadena RAG completa (Retriever -> Prompt -> LLM).

    Returns:
        Una RunnableSequence lista para ser invocada con preguntas del usuario.
    """
    llm_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        llm = OllamaLLM(model=llm_model, base_url=ollama_base_url)
    except Exception as e:
        raise ConnectionError(
             f"Fallo grave de comunicación con Ollama ({ollama_base_url}): Verifica si el programa matriz de ollama está encendido. Detalles: {e}"
        )

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

    def format_docs(docs) -> str:
         return "\n\n".join(f"[{i+1}] (Fuente: {os.path.basename(doc.metadata.get('source', 'Ley Desconocida'))}) \n{doc.page_content}" for i, doc in enumerate(docs))

    # Construimos la red (chain) de la Cadena RAG:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def query_legal(question: str) -> str:
    """Función de alto nivel para enviar de manera simplificada una pregunta a la Ley Chilena.

    Args:
        question: La instrucción o pregunta semántica de corte legal a resolver.

    Returns:
        El texto legal devuelto por el motor generativo tras un escaneo en ChromaDB y su razonamiento con Llama 3.1.
    """
    print(f"Buscando en la base documental y procesando razonamiento para: '{question}'...")
    try:
        chain = get_rag_chain()
        result = chain.invoke(question)
        return result # type: ignore
    except Exception as e:
         return f"❌ Error interno durante el flujo de Retrieval/RAG: {e}"
