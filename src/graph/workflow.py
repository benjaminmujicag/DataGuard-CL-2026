"""LangGraph state machine: schema analysis, classification, RAG, report."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.tools import extract_schema_columns, _classify_column_dict
from src.retrieval.retriever import query_legal

GRAPH_NODE_ORDER: tuple[str, ...] = (
    "analyze_schema",
    "classify_columns",
    "query_rag",
    "evaluate_risk",
    "generate_report",
)

GRAPH_NODE_LABELS_ES: dict[str, str] = {
    "analyze_schema": "Leer DDL y extraer columnas (solo metadatos)",
    "classify_columns": "Clasificar columnas por reglas de riesgo",
    "query_rag": "Consultar ley en RAG (riesgo medio/alto o desconocido)",
    "evaluate_risk": "Fijar mitigaciones sugeridas",
    "generate_report": "Armar reporte ejecutivo",
}

MERMAID_AUDIT_FLOW: str = """
flowchart LR
    A[analyze_schema<br/>Leer DDL] --> B[classify_columns<br/>Clasificar]
    B --> C[query_rag<br/>RAG legal]
    C --> D[evaluate_risk<br/>Mitigar]
    D --> E[generate_report<br/>Reporte]
    E --> F((Fin))
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


def analyze_schema(state: AuditWorkflowState) -> dict[str, Any]:
    """Load DDL and build column list (metadata only)."""
    path = state.get("schema_path", "")
    if not path or not os.path.exists(path):
        err = f"Ruta de esquema inválida o inexistente: {path}"
        return {"errors": state.get("errors", []) + [err], "columns": []}
    try:
        cols = extract_schema_columns(path)
        return {"columns": cols, "errors": state.get("errors", [])}
    except OSError as e:
        return {"errors": state.get("errors", []) + [str(e)], "columns": []}


def classify_columns(state: AuditWorkflowState) -> dict[str, Any]:
    """Apply rule-based categorization per column."""
    if state.get("errors"):
        return {}
    out: list[dict[str, Any]] = []
    for row in state.get("columns", []):
        meta = _classify_column_dict(row["column"], row["table"])
        meta = dict(meta)
        meta.pop("unknown", None)
        out.append({**row, **meta})
    return {"columns": out}


def query_rag(state: AuditWorkflowState) -> dict[str, Any]:
    """Enrich medium/high-risk rows with RAG snippets from the law."""
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


def evaluate_risk(state: AuditWorkflowState) -> dict[str, Any]:
    """Add mitigation hints to each finding."""
    if state.get("errors"):
        return {}
    new_cols: list[dict[str, Any]] = []
    for row in state.get("columns", []):
        r = dict(row)
        r["mitigacion"] = _mitigacion_for_riesgo(str(r.get("riesgo", "Bajo")))
        new_cols.append(r)
    return {"columns": new_cols}


def generate_report(state: AuditWorkflowState) -> dict[str, Any]:
    """Build executive dict: resumen + hallazgos."""
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
    """Compile the audit LangGraph.

    Returns:
        A compiled LangGraph runnable. Invoke with ``.invoke({"schema_path": "..."})``.
    """
    graph = StateGraph(AuditWorkflowState)
    graph.add_node("analyze_schema", analyze_schema)
    graph.add_node("classify_columns", classify_columns)
    graph.add_node("query_rag", query_rag)
    graph.add_node("evaluate_risk", evaluate_risk)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("analyze_schema")
    graph.add_edge("analyze_schema", "classify_columns")
    graph.add_edge("classify_columns", "query_rag")
    graph.add_edge("query_rag", "evaluate_risk")
    graph.add_edge("evaluate_risk", "generate_report")
    graph.add_edge("generate_report", END)
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
            "El grafo LangGraph no entregó un reporte (nodo generate_report). "
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
