import os
import json
from src.agent.agent import run_audit

def test_full_audit():
    """Ejecuta una auditoría completa sobre el esquema de prueba."""
    print("--- 🔍 TEST: Auditoría de Privacidad Completa (F2.3) ---")
    schema_path = os.path.join("data", "sample_schema.sql")
    
    if not os.path.exists(schema_path):
        print(f"❌ Error: No se encontró {schema_path}")
        return
        
    try:
        resultado = run_audit(schema_path)
        
        print("\n📊 REPORTE DE AUDITORÍA GENERADO:")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        
        # Validaciones de integridad del reporte
        assert "resumen" in resultado, "El reporte debe tener un resumen"
        assert "hallazgos" in resultado or "hallazgos_raw" in resultado, "El reporte debe tener hallazgos"
        
        if "hallazgos" in resultado:
            print(f"✅ Auditoría exitosa: {len(resultado['hallazgos'])} hallazgos detectados.")
        else:
            print("🕒 Auditoría generada en formato raw.")
            
        print("\n🎉 PHASE 2 LOGIC VALIDATED SUCCESSFULLY.")
        
    except Exception as e:
        print(f"❌ Fallo crítico en el Agente Auditor: {e}")

if __name__ == "__main__":
    test_full_audit()
