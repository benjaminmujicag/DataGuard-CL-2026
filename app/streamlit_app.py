import json
import os
import sys
import tempfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.agent import run_audit
from src.graph.workflow import (
    GRAPH_NODE_ORDER,
    MERMAID_AUDIT_FLOW,
    iter_graph_audit_steps,
)
from src.retrieval.retriever import query_legal
from src.report.generator import generate_pdf_report

st.set_page_config(
    page_title="CL-DataGuard 2026 | Compliance Hub",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
    [data-testid="stToolbar"] { visibility: visible !important; }
    div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    .block-container { padding-top: 0.75rem; padding-bottom: 2rem; }
    h1 { letter-spacing: -0.02em; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🛡️ CL-DataGuard 2026")
st.caption("Consultor legal local (RAG) y auditoría de metadatos SQL — Ley 19.628 / 21.719")
st.markdown("---")

tab_chat, tab_audit, tab_graph = st.tabs(
    [
        "💬 Consultor Legal (RAG)",
        "🔍 Auditoría de Esquema (Agente)",
        "📊 Auditoría LangGraph",
    ]
)


def _mermaid_chart(diagram: str, height: int = 360) -> None:
    """Render Mermaid via CDN (requiere red). El texto es estático del repo (no inyectar SQL)."""
    body = diagram.strip()
    components.html(
        f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"/></head>
<body style="margin:0;background:transparent;">
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: false, theme: "dark", securityLevel: "loose" }});
    const el = document.getElementById("mg");
    el.removeAttribute("data-processed");
    await mermaid.run({{ nodes: [el] }});
  </script>
  <div id="mg" class="mermaid" style="text-align:center;">{body}</div>
</body></html>
        """,
        height=height,
    )


def _render_audit_tables(reporte: dict) -> None:
    """Muestra métricas, tabla y descargas JSON/PDF comunes a auditorías."""
    res = reporte.get("resumen", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Tablas analizadas", res.get("total_tablas", 0))

    hallazgos = reporte.get("hallazgos", [])
    riesgos_altos = len([h for h in hallazgos if h.get("riesgo") == "Alto"])
    col2.metric("Riesgos altos", riesgos_altos, delta=riesgos_altos, delta_color="inverse")
    col3.metric("Fecha", res.get("fecha_auditoria", str(datetime.now().date())))

    st.markdown("### Detalle de hallazgos")

    if hallazgos:
        display_data = []
        for h in hallazgos:
            emoji = (
                "🔴"
                if h.get("riesgo") == "Alto"
                else "🟡"
                if h.get("riesgo") == "Medio"
                else "🟢"
            )
            display_data.append(
                {
                    "Prioridad": emoji,
                    "Tabla": h.get("tabla"),
                    "Columna": h.get("columna"),
                    "Categoría": h.get("categoria"),
                    "Riesgo": h.get("riesgo"),
                    "Mitigación": h.get("mitigacion"),
                }
            )

        st.dataframe(display_data, use_container_width=True, hide_index=True)

        with st.expander("Ver base legal por columna"):
            for h in hallazgos:
                st.markdown(f"**{h.get('tabla')}.{h.get('columna')}**")
                st.caption(str(h.get("base_legal", ""))[:8000])
                st.divider()

        json_report = json.dumps(reporte, indent=2, ensure_ascii=False)
        st.download_button(
            label="Descargar reporte JSON",
            data=json_report,
            file_name=f"auditoria_dataguard_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            key=f"dl_json_{id(reporte)}",
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            pdf_path = tmp_pdf.name
        try:
            generate_pdf_report(reporte, pdf_path)
            with open(pdf_path, "rb") as pf:
                pdf_bytes = pf.read()
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

        st.download_button(
            label="Descargar reporte PDF",
            data=pdf_bytes,
            file_name=f"auditoria_dataguard_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key=f"dl_pdf_{id(reporte)}",
        )
    else:
        st.info("No se detectaron columnas o hallazgos en este esquema.")


# --- Pestaña 1: chat ---
with tab_chat:
    st.subheader("Asistente jurídico")

    with st.expander("Instrucciones"):
        st.write(
            "Consultas sobre protección de datos en Chile. Las respuestas se apoyan en el corpus "
            "legal ingerido (RAG)."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hola. Soy el consultor legal. ¿En qué puedo ayudarte?",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Escribe tu consulta…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en la base legal…"):
                respuesta = query_legal(prompt)
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})

# pestaña 2
with tab_audit:
    st.subheader("Auditoría con agente (ReAct)")
    st.info("Sube un `.sql` con `CREATE TABLE`. Usa LLM + herramientas.")

    uploaded_file = st.file_uploader(
        "Esquema SQL (modo agente)", type=["sql"], key="upload_agent"
    )

    if uploaded_file is not None and st.button("Iniciar auditoría (agente)", key="btn_agent"):
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sql", mode="wb") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("Agente analizando esquema y ley…"):
                reporte = run_audit(tmp_path)

            if "error" in reporte:
                st.error(str(reporte["error"]))
            else:
                st.success("Auditoría finalizada (agente)")
                _render_audit_tables(reporte)
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

# Pestaña 3 — F3.4 completo: progreso por nodo + Mermaid + descargas
with tab_graph:
    st.subheader("Auditoría con LangGraph")
    st.info(
        "Flujo fijado: analizar esquema → clasificar → RAG → mitigación → reporte. "
        "Debajo verás el avance por nodo y el diagrama del grafo."
    )

    with st.expander("Diagrama del flujo (Mermaid)"):
        st.caption("Requiere conexión al CDN de Mermaid en el navegador.")
        _mermaid_chart(MERMAID_AUDIT_FLOW, height=340)
        st.markdown("**Código fuente del diagrama**")
        st.code(MERMAID_AUDIT_FLOW.strip(), language="text")

    up_g = st.file_uploader("Esquema SQL (LangGraph)", type=["sql"], key="upload_graph")

    if up_g is not None and st.button("Ejecutar flujo LangGraph", key="btn_graph"):
        tmp_sql: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sql", mode="wb") as tf:
                tf.write(up_g.getvalue())
                tmp_sql = tf.name

            progress = st.progress(0, text="Iniciando…")
            reporte: dict = {}
            n_steps = len(GRAPH_NODE_ORDER)

            with st.status("Ejecutando nodos del grafo…", expanded=True) as status_box:
                for i, (node_id, label, delta) in enumerate(iter_graph_audit_steps(tmp_sql)):
                    status_box.write(f"**{node_id}** — {label}")
                    progress.progress(
                        min((i + 1) / n_steps, 1.0),
                        text=f"Paso {i + 1}/{n_steps}: {node_id}",
                    )
                    if isinstance(delta, dict) and "report" in delta:
                        reporte = delta["report"]

            progress.progress(1.0, text="Listo")

            errs = (reporte.get("resumen") or {}).get("errores")
            if errs:
                st.error("Errores: " + "; ".join(str(x) for x in errs))
            else:
                st.success("Auditoría LangGraph finalizada")

            _render_audit_tables(reporte)
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if tmp_sql and os.path.exists(tmp_sql):
                os.remove(tmp_sql)
