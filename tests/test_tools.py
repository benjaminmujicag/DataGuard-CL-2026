import os
from src.agent.tools import read_schema, categorize_column, get_legal_context

def test_read_schema():
    """Test standard SQL schema parsing."""
    print("--- Test: Read Schema ---")
    schema_path = os.path.join("data", "sample_schema.sql")
    if not os.path.exists(schema_path):
        print(f"Error: {schema_path} does not exist.")
        return
        
    result = read_schema.invoke({"file_path": schema_path})
    print(result)
    assert "Table: usuarios" in result
    assert "Table: fichas_clinicas" in result
    assert "rut (VARCHAR(12))" in result
    assert "diagnostico_medico (TEXT)" in result
    print("✅ Read Schema Tool passed.")

def test_categorize_column():
    """Test column categorization rules."""
    print("\n--- Test: Categorize Column ---")
    
    # Test 1: Identificador
    res1 = categorize_column.invoke({"column_name": "rut", "table_name": "usuarios"})
    print(f"RUT -> {res1}")
    assert "Alto" in res1
    
    # Test 2: Contacto
    res2 = categorize_column.invoke({"column_name": "email", "table_name": "usuarios"})
    print(f"Email -> {res2}")
    assert "Medio" in res2
    
    # Test 3: Salud
    res3 = categorize_column.invoke({"column_name": "diagnostico_medico", "table_name": "fichas_clinicas"})
    print(f"Salud -> {res3}")
    assert "Sensibles" in res3
    
    # Test 4: Desconocida (LLM fallback)
    res4 = categorize_column.invoke({"column_name": "obs_interna", "table_name": "configuracion"})
    print(f"OBS -> {res4}")
    assert "Desconocida" in res4
    
    print("✅ Categorize Column Tool passed.")

def test_legal_context():
    """Test connection to RAG from tools."""
    print("\n--- Test: Get Legal Context (RAG) ---")
    query = "¿Qué es un dato biométrico?"
    result = get_legal_context.invoke({"query": query})
    print(result)
    assert "identificación biológica" in result.lower() or "biométrico" in result.lower()
    print("✅ Get Legal Context Tool passed (RAG linked correctly).")

if __name__ == "__main__":
    try:
        test_read_schema()
        test_categorize_column()
        # Nota: test_legal_context requiere que Ollama y ChromaDB estén listos
        test_legal_context()
        print("\n🎉 ALL TOOLS VALIDATED SUCCESSFULLY.")
    except Exception as e:
        print(f"\n❌ Tool Validation Failed: {e}")
