"""LangGraph state machine: schema analysis, classification, RAG, report.

# Review: opus-4.6 · 2026-04-03
# Reviewed post-migration (imports moved from src.agent.tools to src.schema.parse_ddl).
# Architecture: deterministic sequential graph, no LLM-driven branching.

# Updated: opus-4.6 · 2026-04-03
# Removed enrich_context node — classify_column merged into parse_ddl.
# Rationale: with only column name + type available, a separate regex step adds
# no new information. parse_ddl now outputs columns already categorized.
# Graph reduced from 5 to 4 nodes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.schema.parse_ddl import extract_schema_columns, classify_column
from src.retrieval.retriever import query_legal

GRAPH_NODE_ORDER: tuple[str, ...] = (
    "parse_ddl",
    "rag_classify",
    "assign_criticality",
    "emit_report",
)

GRAPH_NODE_LABELS_ES: dict[str, str] = {
    "parse_ddl": "Parsear DDL, extraer columnas y clasificar por heurística (sin LLM)",
    "rag_classify": "Clasificar con RAG + LLM anclado al corpus legal (solo riesgo medio/alto)",
    "assign_criticality": "Asignar mitigaciones por tabla de reglas",
    "emit_report": "Consolidar hallazgos y generar reporte ejecutivo",
}

MERMAID_AUDIT_FLOW: str = """
flowchart LR
    A["parse_ddl<br/>Leer DDL + clasificar"] --> B["rag_classify<br/>RAG + LLM"]
    B --> C["assign_criticality<br/>Mitigaciones"]
    C --> D["emit_report<br/>Reporte"]
    D --> E((Fin))
"""


class AuditWorkflowState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    schema_path: str
    errors: list[str]
    columns: list[dict[str, Any]]
    report: dict[str, Any]


def _mitigacion_for_riesgo(riesgo: str) -> str:
    if riesgo == "Alto":
        return (
            "Cifrado, minimización de datos, evaluación de impacto, controles de acceso "
            "estrictos y registro de accesos."
        )
    if riesgo == "Medio":
        return (
            "Pseudonimización o enmascaramiento donde sea factible; políticas de retención "
            "claras y base legal del tratamiento."
        )
    return "Registro y revisión periódica; conservar proporcionalidad y finalidad."


def parse_ddl(state: AuditWorkflowState) -> dict[str, Any]:
    """Parse DDL file, extract column list and apply heuristic classification.

    Reads the SQL schema, parses CREATE TABLE statements, and immediately
    applies regex-based column classification (category + risk level).
    No LLM call. This merges the former parse_ddl (F4.1) and enrich_context
    (F4.2) steps: since only column name and SQL type are available, there is
    no benefit in splitting parsing from classification into separate nodes.
    """
    path = state.get("schema_path", "")
    if not path or not os.path.exists(path):
        err = f"Ruta de esquema inválida o inexistente: {path}"
        return {"errors": state.get("errors", []) + [err], "columns": []}
    try:
        raw_cols = extract_schema_columns(path)
    except OSError as e:
        return {"errors": state.get("errors", []) + [str(e)], "columns": []}

    enriched: list[dict[str, Any]] = []
    for row in raw_cols:
        meta = classify_column(row["column"], row["table"])
        meta = dict(meta)
        meta.pop("unknown", None)
        enriched.append({**row, **meta})

    return {"columns": enriched, "errors": state.get("errors", [])}


def rag_classify(state: AuditWorkflowState) -> dict[str, Any]:
    """Enrich medium/high-risk columns with RAG context from the legal corpus.

    Only columns flagged as medium/high risk or Desconocida by the heuristic
    layer are sent to the LLM. The prompt is structured and anchored to
    retrieved legal fragments — no free-form reasoning allowed.
    """
    if state.get("errors"):
        return {}
    new_cols: list[dict[str, Any]] = []
    for row in state.get("columns", []):
        r = dict(row)
        riesgo = r.get("riesgo", "Bajo")
        categoria = r.get("categoria", "")
        if riesgo in ("Alto", "Medio") or categoria == "Desconocida":
            q = (
                f"Según la legislación chilena de protección de datos personales (Ley 21.719 "
                f"y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
                f"de datos clasificados como: {categoria}? Responde citando artículos si el "
                f"texto lo permite."
            )
            try:
                r["base_legal"] = query_legal(q)
            except Exception as e:
                r["base_legal"] = f"(Error al consultar RAG: {e})"
        else:
            r["base_legal"] = (
                "No se requirió profundización normativa automática (riesgo bajo según reglas "
                "heurísticas)."
            )
        new_cols.append(r)
    return {"columns": new_cols}


def assign_criticality(state: AuditWorkflowState) -> dict[str, Any]:
    """Map legal category to mitigation actions using explicit rules.

    Applies a deterministic rule table: risk level → required technical actions.
    No additional LLM call.
    """
    if state.get("errors"):
        return {}
    new_cols: list[dict[str, Any]] = []
    for row in state.get("columns", []):
        r = dict(row)
        r["mitigacion"] = _mitigacion_for_riesgo(str(r.get("riesgo", "Bajo")))
        new_cols.append(r)
    return {"columns": new_cols}


def emit_report(state: AuditWorkflowState) -> dict[str, Any]:
    """Consolidate all column findings into the final executive report dict.

    Aggregates per-column results into summary metrics and the hallazgos list.
    This is the last node; its output is the product deliverable.
    """
    if state.get("errors"):
        report = {
            "resumen": {
                "total_tablas": 0,
                "total_columnas": 0,
                "fecha_auditoria": str(date.today()),
                "errores": state.get("errors", []),
            },
            "hallazgos": [],
        }
        return {"report": report}

    cols = state.get("columns", [])
    tables = {c["table"] for c in cols}
    hallazgos: list[dict[str, Any]] = []
    for c in cols:
        hallazgos.append(
            {
                "tabla": c["table"],
                "columna": c["column"],
                "categoria": c.get("categoria", ""),
                "riesgo": c.get("riesgo", ""),
                "base_legal": c.get("base_legal", ""),
                "mitigacion": c.get("mitigacion", ""),
                "sql_type": c.get("sql_type", ""),
            }
        )

    report = {
        "resumen": {
            "total_tablas": len(tables),
            "total_columnas": len(cols),
            "fecha_auditoria": str(date.today()),
            "ley_fuente": "Ley 21.719 / corpus ingerido en ChromaDB",
        },
        "hallazgos": hallazgos,
    }
    return {"report": report}


def create_audit_workflow() -> Any:
    """Compile the deterministic audit LangGraph.

    Nodes are fixed and sequential; no LLM-driven branching. Order:
    parse_ddl → rag_classify → assign_criticality → emit_report.

    Returns:
        A compiled LangGraph runnable. Invoke with ``.invoke({"schema_path": "..."})``.
    """
    graph = StateGraph(AuditWorkflowState)
    graph.add_node("parse_ddl", parse_ddl)
    graph.add_node("rag_classify", rag_classify)
    graph.add_node("assign_criticality", assign_criticality)
    graph.add_node("emit_report", emit_report)

    graph.set_entry_point("parse_ddl")
    graph.add_edge("parse_ddl", "rag_classify")
    graph.add_edge("rag_classify", "assign_criticality")
    graph.add_edge("assign_criticality", "emit_report")
    graph.add_edge("emit_report", END)
    return graph.compile()


def run_graph_audit_traced(
    schema_path: str,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Run the workflow streaming node updates for UI progress.

    Args:
        schema_path: Path to a SQL DDL file.

    Returns:
        Tuple of ``(report, step_log)`` where ``step_log`` is
        ``[(node_id, label_es), ...]`` in execution order.
    """
    app = create_audit_workflow()
    initial: AuditWorkflowState = {
        "schema_path": schema_path,
        "errors": [],
        "columns": [],
    }
    report: dict[str, Any] = {}
    step_log: list[tuple[str, str]] = []

    for chunk in app.stream(initial):
        for node_id, delta in chunk.items():
            label = GRAPH_NODE_LABELS_ES.get(node_id, node_id)
            step_log.append((node_id, label))
            if isinstance(delta, dict) and "report" in delta:
                report = delta["report"]

    if not report:
        raise RuntimeError(
            "El grafo LangGraph no entregó un reporte (nodo emit_report). "
            "Revisa dependencias y el esquema SQL de entrada."
        )

    return report, step_log


def iter_graph_audit_steps(schema_path: str) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(node_id, label_es, delta)`` after each node completes (for live UIs)."""
    app = create_audit_workflow()
    initial: AuditWorkflowState = {
        "schema_path": schema_path,
        "errors": [],
        "columns": [],
    }
    for chunk in app.stream(initial):
        for node_id, delta in chunk.items():
            label = GRAPH_NODE_LABELS_ES.get(node_id, node_id)
            yield node_id, label, delta if isinstance(delta, dict) else {}


def run_graph_audit(schema_path: str) -> dict[str, Any]:
    """Run the workflow and return only the report payload.

    Args:
        schema_path: Path to a SQL DDL file.

    Returns:
        Dict with keys ``resumen`` and ``hallazgos`` (or error info inside ``resumen``).
    """
    report, _ = run_graph_audit_traced(schema_path)
    return report
