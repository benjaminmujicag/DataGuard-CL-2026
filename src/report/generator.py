"""Generate structured JSON and PDF reports from audit payloads.

# Review: opus-4.6 · 2026-04-03
# Code is clean: proper UTF-8 handling, Unicode font fallback, ReportLab usage correct.
# Minor note: PDF table truncates mitigacion to 200 chars (acceptable for layout). Approved.
"""

from __future__ import annotations

import json
import os
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_body_font() -> str:
    """Register a Unicode TTF when available; fall back to Helvetica."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    cache_attr = "_cl_dataguard_body_font"
    if getattr(_register_body_font, cache_attr, None):
        return getattr(_register_body_font, cache_attr)

    candidates: list[str] = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates.append(os.path.join(windir, "Fonts", "arial.ttf"))
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            pdfmetrics.registerFont(TTFont("AuditBody", path))
            setattr(_register_body_font, cache_attr, "AuditBody")
            return "AuditBody"
    setattr(_register_body_font, cache_attr, "Helvetica")
    return "Helvetica"


def generate_json_report(audit_result: dict[str, Any], output_path: str) -> str:
    """Write ``audit_result`` as UTF-8 JSON.

    Args:
        audit_result: Payload shaped like ``run_graph_audit`` / agent JSON output.
        output_path: Destination path for the file.

    Returns:
        The ``output_path`` written.
    """
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, ensure_ascii=False, indent=2)
    return output_path


def generate_pdf_report(audit_result: dict[str, Any], output_path: str) -> str:
    """Build a Spanish executive PDF (summary + findings table).

    Args:
        audit_result: Same shape as ``generate_json_report``.
        output_path: Destination ``.pdf`` path.

    Returns:
        The ``output_path`` written.
    """
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    font = _register_body_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        name="BodyEs",
        parent=styles["Normal"],
        fontName=font,
        fontSize=10,
        leading=14,
    )
    heading = ParagraphStyle(
        name="HeadingEs",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=14,
        leading=18,
        spaceAfter=12,
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    story: list[Any] = []

    story.append(Paragraph(escape("CL-DataGuard 2026 — Reporte de auditoría"), heading))
    resumen = audit_result.get("resumen") or {}
    fecha = escape(str(resumen.get("fecha_auditoria", "")))
    n_tab = resumen.get("total_tablas", 0)
    n_col = resumen.get("total_columnas", 0)
    story.append(
        Paragraph(
            escape(
                f"Fecha: {fecha}. Tablas: {n_tab}. Columnas analizadas: {n_col}."
            ),
            body,
        )
    )
    ley = resumen.get("ley_fuente")
    if ley:
        story.append(Paragraph(escape(str(ley)), body))
    errs = resumen.get("errores")
    if errs:
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph(escape("Errores durante la auditoría:"), heading)
        )
        for e in errs:
            story.append(Paragraph(escape(str(e)), body))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(escape("Detalle de hallazgos"), heading))

    hallazgos = audit_result.get("hallazgos") or []
    if not hallazgos:
        story.append(Paragraph(escape("Sin hallazgos registrados."), body))
    else:
        table_data: list[list[Any]] = [
            [
                Paragraph(escape("Tabla"), body),
                Paragraph(escape("Columna"), body),
                Paragraph(escape("Categoría"), body),
                Paragraph(escape("Riesgo"), body),
                Paragraph(escape("Mitigación (resumen)"), body),
            ],
        ]
        for h in hallazgos:
            mit = str(h.get("mitigacion", ""))[:200]
            table_data.append(
                [
                    Paragraph(escape(str(h.get("tabla", ""))), body),
                    Paragraph(escape(str(h.get("columna", ""))), body),
                    Paragraph(escape(str(h.get("categoria", ""))), body),
                    Paragraph(escape(str(h.get("riesgo", ""))), body),
                    Paragraph(escape(mit), body),
                ]
            )
        tbl = Table(
            table_data,
            colWidths=[3 * cm, 3 * cm, 3.2 * cm, 2 * cm, 5.5 * cm],
            repeatRows=1,
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(tbl)

    doc.build(story)
    return output_path
