import os
from dotenv import load_dotenv
from src.retrieval.retriever import query_legal

# Override langsmith para evitar cuelgues si no hay Api key configurada
os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

print("--- TEST 1: Elemento legal explícito ---")
res1 = query_legal("¿Qué es un dato personal según la ley?")
print(f"Respuesta:\n{res1}\n")

print("--- TEST 2: Elemento fuera de contexto legal ---")
res2 = query_legal("¿Cuál es la capital de Francia?")
print(f"Respuesta:\n{res2}\n")

print("--- TEST 3: Sub-tipo de datos explícito ---")
res3 = query_legal("¿Qué son los datos sensibles y qué protección especial tienen?")
print(f"Respuesta:\n{res3}\n")
