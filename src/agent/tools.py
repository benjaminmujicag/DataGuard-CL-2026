import os
import re
from typing import Any

from langchain_core.tools import tool
from langsmith import traceable

from src.retrieval.retriever import query_legal


def _parse_ddl_tables(content: str) -> list[tuple[str, str, str]]:
    """Extract (table_name, column_name, sql_type) tuples from DDL text.

    Args:
        content: Full SQL file contents.

    Returns:
        List of tuples for each declared column (skips PRIMARY/FOREIGN KEY lines).
    """
    table_matches = re.finditer(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\);",
        content,
        re.S | re.I,
    )
    rows: list[tuple[str, str, str]] = []
    sql_key_prefixes = ("PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "KEY", "CHECK")
    for match in table_matches:
        table_name = match.group(1)
        table_body = match.group(2)
        for line in table_body.strip().split("\n"):
            line_stripped = line.strip().replace(",", "")
            if not line_stripped:
                continue
            first_tok = line_stripped.split()[0].upper()
            if first_tok in sql_key_prefixes:
                continue
            parts = line_stripped.split()
            if not parts:
                continue
            col_name = parts[0]
            col_type = parts[1] if len(parts) > 1 else "UNKNOWN"
            rows.append((table_name, col_name, col_type))
    return rows


def extract_schema_columns(file_path: str) -> list[dict[str, str]]:
    """Load DDL file and return structured column metadata (no SQL execution).

    Args:
        file_path: Path to the .sql file.

    Returns:
        One dict per column with keys: table, column, sql_type.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [
        {"table": t, "column": c, "sql_type": typ}
        for t, c, typ in _parse_ddl_tables(content)
    ]


@traceable(name="read_schema", run_type="tool")
def _read_schema_core(file_path: str) -> str:
    """Implementation of read_schema; traced as a LangSmith tool run."""
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        grouped: dict[str, list[str]] = {}
        for table_name, col_name, col_type in _parse_ddl_tables(content):
            grouped.setdefault(table_name, []).append(f"  - {col_name} ({col_type})")
        if not grouped:
            return (
                "No tables found in the SQL file. Ensure it contains valid "
                "CREATE TABLE statements."
            )
        blocks = [f"Table: {tbl}\n" + "\n".join(lines) for tbl, lines in grouped.items()]
        return "\n\n".join(blocks)
    except Exception as e:
        return f"Error processing SQL file: {str(e)}"


@tool
def read_schema(file_path: str) -> str:
    """Reads a SQL DDL file (.sql) and extracts names of tables and columns.

    Use this tool to understand the database structure before auditing.

    Args:
        file_path: Absolute or relative path to the .sql file.

    Returns:
        A text representation of the schema (Tables and their columns).
    """
    return _read_schema_core(file_path)


def _classify_column_dict(column_name: str, table_name: str) -> dict[str, str]:
    """Rule-based classification for a single column."""
    col = column_name.lower().strip()

    if any(k in col for k in ["rut", "cedula", "dni", "passport", "pasaporte", "identificador"]):
        return {
            "categoria": "Datos identificadores",
            "riesgo": "Alto",
            "accion": "Citar Ley 21.719 Art. sobre identificación.",
        }

    if any(
        k in col
        for k in [
            "diagnostico",
            "clinica",
            "ficha",
            "medico",
            "enfermedad",
            "tratamiento",
            "salud",
        ]
    ):
        return {
            "categoria": "Datos de salud (Sensibles)",
            "riesgo": "Alto",
            "accion": "Citar Ley 21.719 sobre datos sensibles de salud.",
        }

    if any(
        k in col
        for k in [
            "sueldo",
            "renta",
            "salario",
            "cuenta_banco",
            "tarjeta",
            "credit_card",
            "financiero",
        ]
    ):
        return {
            "categoria": "Datos financieros",
            "riesgo": "Alto",
            "accion": "Requiere evaluación de impacto financiero.",
        }

    if any(
        k in col
        for k in ["email", "correo", "telefono", "phone", "direccion", "address", "celular"]
    ):
        return {
            "categoria": "Datos de contacto",
            "riesgo": "Medio",
            "accion": "Verificar consentimiento de uso.",
        }

    if any(k in col for k in ["huella", "biometric", "facial", "iris", "voz", "adn"]):
        return {
            "categoria": "Datos biométricos",
            "riesgo": "Alto",
            "accion": "Prohibición general salvo excepciones legales explícitas.",
        }

    if any(k in col for k in ["edad", "birth", "nacimiento", "es_menor"]) and "user" in table_name.lower():
        return {
            "categoria": "Potencial dato de menores",
            "riesgo": "Alto",
            "accion": "Aplicar principio de interés superior del niño.",
        }

    if any(k in col for k in ["latitud", "longitud", "gps", "ubicacion", "location"]):
        return {
            "categoria": "Datos de geolocalización",
            "riesgo": "Medio",
            "accion": "Verificar necesidad de tratamiento proporcional.",
        }

    if any(k in col for k in ["ip_address", "user_agent", "log", "session", "id_sesion"]):
        return {
            "categoria": "Logs y auditoría",
            "riesgo": "Bajo",
            "accion": "Mantener registro de acceso.",
        }

    return {
        "categoria": "Desconocida",
        "riesgo": "Medio",
        "accion": "Requiere análisis de contexto adicional y consulta legal.",
        "unknown": True,
    }


@traceable(name="categorize_column", run_type="tool")
def _categorize_column_core(column_name: str, table_name: str) -> str:
    """Implementation of categorize_column; traced as a LangSmith tool run."""
    d = _classify_column_dict(column_name, table_name)
    if d.get("unknown"):
        return (
            f"Categoría: Desconocida (Requiere análisis de contexto LLM) | "
            f"Columna: {column_name} | Tabla: {table_name}"
        )
    return (
        f"Categoría: {d['categoria']} | Riesgo: {d['riesgo']} | Acción: {d['accion']}"
    )


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
    return _categorize_column_core(column_name, table_name)


@traceable(name="get_legal_context", run_type="tool")
def _get_legal_context_core(query: str) -> str:
    """Implementation of get_legal_context; traced as a LangSmith tool run."""
    try:
        result = query_legal(query)
        return f"Legal Scan Result for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error querying local law knowledge: {str(e)}"


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
    return _get_legal_context_core(query)
