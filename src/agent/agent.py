import os
import json
from datetime import datetime
from typing import Any, Dict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from src.agent.tools import read_schema, categorize_column, get_legal_context

def create_audit_agent() -> AgentExecutor:
    """Configura y crea el Agente de Auditoría ReAct.
    
    Returns:
        AgentExecutor listo para procesar esquemas SQL.
    """
    llm_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    llm = OllamaLLM(model=llm_model, base_url=ollama_base_url, temperature=0)
    
    tools = [read_schema, categorize_column, get_legal_context]
    
    template = """Eres 'DataGuard-Auditor', un experto en Ciberseguridad y Cumplimiento Legal 
especializado en la Ley 21.719 y Ley 19.628 de Chile.

Tu objetivo es auditar esquemas de bases de datos SQL para identificar riesgos de privacidad.

REGLAS DE ORO:
1. SOLO METADATA: Nunca pidas ni asumas que puedes ver datos reales de filas (SELECT *). Solo nombres de columnas y tablas.
2. RIGOR LEGAL: Siempre que identifiques un riesgo, usa 'get_legal_context' para citar un artículo real.
3. MITIGACIÓN TÉCNICA: Para riesgos Medio o Alto, debes sugerir una medida técnica (Encriptación, Hash, Enmascaramiento, etc.).
4. PENSAMIENTO PASO A PASO: Analiza tabla por tabla y columna por columna.

PROCESO:
1. Usa 'read_schema' para obtener la estructura técnica.
2. Para CADA tabla y columna encontrada, usa 'categorize_column'.
3. Si el resultado es 'Desconocida' o el riesgo es 'Medio/Alto', BUSCA en la ley usando 'get_legal_context'.
4. Consolida todo en un reporte estructurado final.

TOOLS DISPONIBLES:
{tools}

NOMBRES DE HERRAMIENTAS:
{tool_names}

FORMATO DE RESPUESTA:
Debes seguir el formato de pensamiento ReAct:
Thought: [Tu razonamiento]
Action: [Nombre de la herramienta]
Action Input: [El input para la herramienta]
Observation: [Resultado de la herramienta]
... (repetir según sea necesario)
Thought: Ya tengo toda la información para el reporte final.
Final Answer: [Tu reporte estructurado final en formato JSON]

El JSON de la 'Final Answer' debe tener la siguiente estructura exacta:
{{
  "resumen": {{
    "total_tablas": 0,
    "total_columnas": 0,
    "fecha_auditoria": "YYYY-MM-DD"
  }},
  "hallazgos": [
    {{
      "tabla": "nombre_tabla",
      "columna": "nombre_columna",
      "categoria": "ej: Salud",
      "riesgo": "Bajo/Medio/Alto",
      "base_legal": "Cita del artículo",
      "mitigacion": "Sugerencia técnica"
    }}
  ]
}}

Comienza el análisis del archivo: {input}

{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=15  # Aumentamos para permitir análisis profundo
    )

def run_audit(schema_path: str) -> Dict[str, Any]:
    """Ejecuta la auditoría completa sobre un archivo SQL.
    
    Args:
        schema_path: Ruta al archivo .sql
        
    Returns:
        Diccionario con el reporte de auditoría.
    """
    if not os.path.exists(schema_path):
        return {"error": f"No se encontró el archivo de esquema en {schema_path}"}
        
    agent_executor = create_audit_agent()
    
    print(f"🚀 Iniciando proceso de auditoría para: {schema_path}")
    try:
        response = agent_executor.invoke({"input": schema_path})
        final_answer = response.get("output", "")
        
        # Intentamos extraer el JSON de la respuesta final
        try:
            # Los LLMs a veces envuelven el JSON en bloques de código
            if "```json" in final_answer:
                json_str = final_answer.split("```json")[1].split("```")[0].strip()
            elif "```" in final_answer:
                json_str = final_answer.split("```")[1].split("```")[0].strip()
            else:
                json_str = final_answer.strip()
                
            report = json.loads(json_str)
            return report
        except Exception:
            # Fallback si el JSON falla: retornar el texto plano
            return {
                "resumen": {"raw_output": True, "fecha_auditoria": str(datetime.now().date())},
                "hallazgos_raw": final_answer
            }
            
    except Exception as e:
        return {"error": f"Fallo en la ejecución del Agente: {str(e)}"}

if __name__ == "__main__":
    # Prueba manual rápida
    from dotenv import load_dotenv
    load_dotenv()
    
    path = os.path.join("data", "sample_schema.sql")
    reporte = run_audit(path)
    print("\n--- REPORTE GENERADO ---")
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
