import streamlit as st
import sys
import os
import json
import tempfile
from datetime import datetime

# Asegurar que el directorio raíz está en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.retriever import query_legal
from src.agent.agent import run_audit

# --- Configuración de la Página ---
st.set_page_config(
    page_title="CL-DataGuard 2026 | Compliance Hub",
    page_icon="🛡️",
    layout="wide"
)

# --- Header Principal ---
st.title("🛡️ CL-DataGuard 2026")
st.markdown("---")

# --- División por Pestañas ---
tab_chat, tab_audit = st.tabs(["💬 Consultor Legal (RAG)", "🔍 Auditoría de Esquema (Agente)"])

# ==========================================
# PESTAÑA 1: CONSULTOR LEGAL (CHAT)
# ==========================================
with tab_chat:
    st.subheader("Asistente Jurídico — Ley 19.628 & 21.719")
    
    with st.expander("ℹ️ Instrucciones"):
        st.write("Haz consultas sobre la ley de protección de datos personales de Chile. El sistema citará fragmentos legales exactos.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola. Soy el Consultor Legal. ¿En qué puedo ayudarte hoy respecto a la normativa de datos personales?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Escribe tu duda legal aquí...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando bases legales..."):
                respuesta = query_legal(prompt)
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})

# ==========================================
# PESTAÑA 2: AUDITORÍA DE ESQUEMA
# ==========================================
with tab_audit:
    st.subheader("Auditoría Técnica Automatizada")
    st.info("Sube un archivo .sql con tus sentencias 'CREATE TABLE' para identificar riesgos de privacidad.")

    uploaded_file = st.file_uploader("Selecciona tu esquema SQL", type=["sql"])

    if uploaded_file is not None:
        if st.button("🚀 Iniciar Auditoría de Cumplimiento"):
            # Crear un archivo temporal físico para que el Agente pueda leerlo
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sql", mode='wb') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("El Agente Auditor está analizando el esquema y consultando la ley..."):
                try:
                    reporte = run_audit(tmp_path)
                    
                    if "error" in reporte:
                        st.error(f"Error en la auditoría: {reporte['error']}")
                    else:
                        # --- Visualización de Resultados ---
                        st.success("Auditoría Finalizada con Éxito")
                        
                        # Métricas resumidas
                        res = reporte.get("resumen", {})
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Tablas Analizadas", res.get("total_tablas", 0))
                        
                        hallazgos = reporte.get("hallazgos", [])
                        riesgos_altos = len([h for h in hallazgos if h.get("riesgo") == "Alto"])
                        col2.metric("Riesgos ALTOS", riesgos_altos, delta=riesgos_altos, delta_color="inverse")
                        col3.metric("Fecha", res.get("fecha_auditoria", str(datetime.now().date())))

                        # Tabla de Hallazgos
                        st.markdown("### 📋 Detalle de Hallazgos")
                        
                        if hallazgos:
                            # Formatear para tabla legible
                            display_data = []
                            for h in hallazgos:
                                # Determinar emoji por riesgo
                                emoji = "🔴" if h.get("riesgo") == "Alto" else "🟡" if h.get("riesgo") == "Medio" else "🟢"
                                display_data.append({
                                    "Prioridad": emoji,
                                    "Tabla": h.get("tabla"),
                                    "Columna": h.get("columna"),
                                    "Categoría": h.get("categoria"),
                                    "Riesgo": h.get("riesgo"),
                                    "Mitigación": h.get("mitigacion")
                                })
                            
                            st.table(display_data)
                            
                            # Expanders para detalle legal profundo
                            with st.expander("🔍 Ver Base Legal y Razonamiento"):
                                for h in hallazgos:
                                    st.markdown(f"**{h.get('tabla')}.{h.get('columna')}**")
                                    st.caption(f"Base Legal: {h.get('base_legal')}")
                                    st.divider()
                                    
                            # Botón de Descarga JSON
                            json_report = json.dumps(reporte, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Descargar Reporte JSON",
                                data=json_report,
                                file_name=f"auditoria_dataguard_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json"
                            )
                        else:
                            st.balloons()
                            st.info("No se detectaron riesgos de privacidad significativos en este esquema.")

                except Exception as e:
                    st.error(f"Error inesperado: {e}")
                finally:
                    # Limpieza del archivo temporal
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
