"""Parse SQL DDL files and classify columns by data-protection risk.

Extracted from the legacy ``src/agent/tools.py`` module so that the
deterministic LangGraph workflow (``src/graph/workflow.py``) can use
these utilities without pulling in the deprecated AgentExecutor stack.
"""

from __future__ import annotations

import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# DDL parsing
# ---------------------------------------------------------------------------

def _parse_ddl_tables(content: str) -> list[tuple[str, str, str]]:
    """Extract ``(table_name, column_name, sql_type)`` tuples from DDL text.

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
    """Load a DDL file and return structured column metadata (no SQL execution).

    Args:
        file_path: Path to the ``.sql`` file.

    Returns:
        One dict per column with keys: ``table``, ``column``, ``sql_type``.

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


# ---------------------------------------------------------------------------
# Heuristic column classification (regex, no LLM)
# ---------------------------------------------------------------------------

def classify_column(column_name: str, table_name: str) -> dict[str, Any]:
    """Rule-based classification of a single database column.

    Uses naming patterns to assign a privacy category and risk level
    according to Chilean data-protection law (Ley 21.719 / 19.628).

    Args:
        column_name: Name of the database column (e.g. ``rut``, ``email``).
        table_name: Name of the table the column belongs to.

    Returns:
        Dict with keys ``categoria``, ``riesgo``, ``accion`` and optionally
        ``unknown`` (True when the heuristic cannot determine a category).
    """
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
            "diagnostico", "clinica", "ficha", "medico",
            "enfermedad", "tratamiento", "salud",
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
            "sueldo", "renta", "salario", "cuenta_banco",
            "tarjeta", "credit_card", "financiero",
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
