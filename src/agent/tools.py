import re
import os
from langchain_core.tools import tool
from src.retrieval.retriever import query_legal

@tool
def read_schema(file_path: str) -> str:
    """Reads a SQL DDL file (.sql) and extracts names of tables and columns.
    
    Use this tool to understand the database structure before auditing.
    
    Args:
        file_path: Absolute or relative path to the .sql file.
        
    Returns:
        A text representation of the schema (Tables and their columns).
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Regex básica para capturar bloques CREATE TABLE (ignorando mayúsculas)
        # Captura el nombre de la tabla y el contenido entre paréntesis
        table_matches = re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\);", content, re.S | re.I)
        
        results = []
        for match in table_matches:
            table_name = match.group(1)
            table_body = match.group(2)
            
            # Extraer nombres de columnas (asumiendo que empiezan la línea tras espacios/comas)
            # Buscamos palabras al inicio de las líneas que no sean palabras clave SQL
            columns = []
            for line in table_body.strip().split("\n"):
                line = line.strip().replace(",", "")
                if not line or any(line.split()[0].upper() in ["PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "KEY", "CHECK"] for word in [line.split()[0]] if word):
                    continue
                
                parts = line.split()
                if parts:
                    col_name = parts[0]
                    col_type = parts[1] if len(parts) > 1 else "UNKNOWN"
                    columns.append(f"  - {col_name} ({col_type})")
            
            results.append(f"Table: {table_name}\n" + "\n".join(columns))

        if not results:
            return "No tables found in the SQL file. Ensure it contains valid CREATE TABLE statements."
            
        return "\n\n".join(results)

    except Exception as e:
        return f"Error processing SQL file: {str(e)}"

@tool
def categorize_column(column_name: str, table_name: str) -> str:
    """Provides a preliminary legal categorization of a column based on its name.
    
    Uses matching rules from the CL-DataGuard 2026 technical audit manual.
    
    Args:
        column_name: Name of the database column (e.g., 'rut', 'email').
        table_name: Name of the table it belongs to.
        
    Returns:
        A string with the category and risk level (e.g., 'Datos Identificadores - Riesgo Alto').
    """
    col = column_name.lower().strip()
    
    # 1. Datos Identificadores (Riesgo Alto)
    if any(k in col for k in ["rut", "cedula", "dni", "passport", "pasaporte", "identificador"]):
        return "Categoría: Datos identificadores | Riesgo: Alto | Acción: Citar Ley 21.719 Art. sobre identificación."

    # 2. Datos de Salud (Riesgo Alto)
    if any(k in col for k in ["diagnostico", "clinica", "ficha", "medico", "enfermedad", "tratamiento", "salud"]):
        return "Categoría: Datos de salud (Sensibles) | Riesgo: Alto | Acción: Citar Ley 21.719 sobre datos sensibles de salud."

    # 3. Datos Financieros (Riesgo Alto)
    if any(k in col for k in ["sueldo", "renta", "salario", "cuenta_banco", "tarjeta", "credit_card", "financiero"]):
        return "Categoría: Datos financieros | Riesgo: Alto | Acción: Requiere evaluación de impacto financiero."

    # 4. Datos de Contacto (Riesgo Medio)
    if any(k in col for k in ["email", "correo", "telefono", "phone", "direccion", "address", "celular"]):
        return "Categoría: Datos de contacto | Riesgo: Medio | Acción: Verificar consentimiento de uso."

    # 5. Datos Biométricos (Riesgo Alto)
    if any(k in col for k in ["huella", "biometric", "facial", "iris", "voz", "adn"]):
        return "Categoría: Datos biométricos | Riesgo: Alto | Acción: Prohibición general salvo excepciones legales explícitas."

    # 6. Datos de Menores (Riesgo Alto)
    if any(k in col for k in ["edad", "birth", "nacimiento", "es_menor"]) and "user" in table_name.lower():
        # Heurística: si hay edad en una tabla de usuarios, sospechar de menores
        return "Categoría: Potencial dato de menores | Riesgo: Alto | Acción: Aplicar principio de interés superior del niño."

    # 7. Geolocalización (Riesgo Medio)
    if any(k in col for k in ["latitud", "longitud", "gps", "ubicacion", "location"]):
        return "Categoría: Datos de geolocalización | Riesgo: Medio | Acción: Verificar necesidad de tratamiento proporcional."

    # 8. Logs (Riesgo Bajo)
    if any(k in col for k in ["ip_address", "user_agent", "log", "session", "id_sesion"]):
        return "Categoría: Logs y auditoría | Riesgo: Bajo | Acción: Mantener registro de acceso."

    # Si no hay match de reglas, el LLM deberá razonar el resto
    return f"Categoría: Desconocida (Requiere análisis de contexto LLM) | Columna: {column_name} | Tabla: {table_name}"

@tool
def get_legal_context(query: str) -> str:
    """Queries the Chilean Data Protection Law (RAG) to find specific regulations.
    
    Use this tool when you need to confirm if a specific data treatment is allowed 
    or to find the exact penalty for a risk identified.
    
    Args:
        query: The specific question or term to search in the law.
        
    Returns:
        Fragments of the law and legal reasoning.
    """
    try:
        result = query_legal(query)
        return f"Legal Scan Result for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error querying local law knowledge: {str(e)}"
