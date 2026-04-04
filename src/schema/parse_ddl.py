"""Parse SQL DDL files and classify columns by data-protection risk.

Uses ``sqlglot`` for robust DDL parsing (handles schema.table, quoted identifiers,
multi-token types, inline comments, etc.) instead of fragile hand-written regex.

Debilidad #1 fix: sqlglot replaces the regex parser.
Debilidad #2 fix: classify_column now uses sql_type and constraints as secondary signals.
"""

from __future__ import annotations

import os
import re
from typing import Any

import sqlglot
import sqlglot.expressions as exp


# ---------------------------------------------------------------------------
# DDL parsing — sqlglot-based (replaces fragile regex)
# ---------------------------------------------------------------------------

def _extract_constraints(col_def: exp.ColumnDef) -> list[str]:
    """Return a list of constraint keywords for a column definition.

    Args:
        col_def: A sqlglot ColumnDef expression.

    Returns:
        List of uppercase constraint strings, e.g. ``["NOT NULL", "UNIQUE"]``.
    """
    constraints: list[str] = []
    for constraint in col_def.find_all(exp.ColumnConstraint):
        kind = constraint.args.get("kind")
        if kind is None:
            continue
        kind_str = kind.__class__.__name__
        if "NotNull" in kind_str:
            constraints.append("NOT NULL")
        elif "Unique" in kind_str:
            constraints.append("UNIQUE")
        elif "PrimaryKey" in kind_str:
            constraints.append("PRIMARY KEY")
        elif "Default" in kind_str:
            constraints.append("DEFAULT")
    return constraints


def _parse_ddl_tables(content: str) -> list[tuple[str, str, str, list[str]]]:
    """Extract ``(table_name, column_name, sql_type, constraints)`` tuples via sqlglot.

    Args:
        content: Full SQL DDL file contents.

    Returns:
        List of 4-tuples for each declared column.
        ``sql_type`` is the canonical type string (e.g. ``"VARCHAR(255)"``).
        ``constraints`` is a list of constraint keywords (may be empty).
    """
    rows: list[tuple[str, str, str, list[str]]] = []
    try:
        statements = sqlglot.parse(content, error_level=sqlglot.ErrorLevel.WARN)
    except Exception:
        # Fallback: return empty; caller will propagate gracefully.
        return rows

    for statement in statements:
        if statement is None:
            continue
        create = statement if isinstance(statement, exp.Create) else None
        if create is None:
            continue
        # Only process CREATE TABLE statements.
        if create.args.get("kind", "").upper() != "TABLE":
            continue

        table_expr = create.find(exp.Table)
        if table_expr is None:
            continue
        # Support schema.table — prefer table name only.
        table_name: str = table_expr.name or str(table_expr)

        schema_expr = create.find(exp.Schema)
        if schema_expr is None:
            continue

        for col_def in schema_expr.find_all(exp.ColumnDef):
            col_name: str = col_def.name
            dtype = col_def.args.get("kind")
            if dtype is None:
                sql_type = "UNKNOWN"
            else:
                sql_type = dtype.sql(dialect="mysql").upper()
            constraints = _extract_constraints(col_def)
            rows.append((table_name, col_name, sql_type, constraints))

    return rows


def extract_schema_columns(file_path: str) -> list[dict[str, Any]]:
    """Load a DDL file and return structured column metadata (no SQL execution).

    Args:
        file_path: Path to the ``.sql`` file.

    Returns:
        One dict per column with keys: ``table``, ``column``, ``sql_type``,
        ``constraints`` (list of strings).

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    rows = _parse_ddl_tables(content)
    return [
        {
            "table": t,
            "column": c,
            "sql_type": typ,
            "constraints": constr,
        }
        for t, c, typ, constr in rows
    ]


# ---------------------------------------------------------------------------
# Secondary signals: sql_type + constraints (Debilidad #2)
# ---------------------------------------------------------------------------

def _detect_type_signals(sql_type: str, constraints: list[str]) -> dict[str, Any]:
    """Derive privacy hints purely from the SQL type and constraints.

    Args:
        sql_type: Canonical SQL type string, e.g. ``"CHAR(12)"``, ``"BOOLEAN"``.
        constraints: List of constraint keywords (``"UNIQUE"``, ``"NOT NULL"`` …).

    Returns:
        Dict with optional override keys: ``categoria``, ``riesgo``.
        Empty dict if no signal fires.
    """
    s = sql_type.upper()

    # CHAR(9-12) or VARCHAR(9-12) → typical Chilean RUT/cedula width
    m = re.match(r"(?:CHAR|CHARACTER|VARCHAR|CHARACTER VARYING)\((\d+)\)", s)
    if m:
        width = int(m.group(1))
        if 8 <= width <= 13 and "UNIQUE" in constraints:
            return {"categoria": "Datos identificadores", "riesgo": "Alto"}

    # BOOLEAN / TINYINT(1) with UNIQUE NOT NULL → flag column (e.g. es_menor)
    if s in ("BOOLEAN", "BOOL", "TINYINT(1)") and "NOT NULL" in constraints:
        return {}  # Not enough alone; let name heuristic decide.

    return {}


# ---------------------------------------------------------------------------
# Heuristic column classification (regex + type signals, no LLM)
# ---------------------------------------------------------------------------

def classify_column(
    column_name: str,
    table_name: str,
    sql_type: str = "",
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    """Rule-based classification of a single database column.

    Uses naming patterns as primary signal, then sql_type and constraints as
    secondary signals to disambiguate generic column names.

    Args:
        column_name: Name of the database column (e.g. ``rut``, ``email``).
        table_name:  Name of the table the column belongs to.
        sql_type:    SQL type string from DDL (e.g. ``"CHAR(12)"``). Optional.
        constraints: List of constraint keywords (e.g. ``["UNIQUE", "NOT NULL"]``). Optional.

    Returns:
        Dict with keys ``categoria``, ``riesgo``, ``accion`` and optionally
        ``unknown`` (True when the heuristic cannot determine a category).
    """
    col = column_name.lower().strip()
    tbl = table_name.lower().strip()
    constr = constraints or []

    # --- Primary signals: column name ---

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

    if any(k in col for k in ["edad", "birth", "nacimiento", "es_menor"]) and "user" in tbl:
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

    # --- Secondary signals: sql_type + constraints (Debilidad #2) ---
    if sql_type:
        type_hint = _detect_type_signals(sql_type, constr)
        if type_hint:
            base: dict[str, Any] = {
                "accion": "Citar Ley 21.719 Art. sobre identificación (deducido del tipo SQL).",
            }
            base.update(type_hint)
            return base

    # --- Fallback ---
    return {
        "categoria": "Desconocida",
        "riesgo": "Medio",
        "accion": "Requiere análisis de contexto adicional y consulta legal.",
        "unknown": True,
    }
