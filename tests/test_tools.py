"""Tests for DDL parsing and column classification (src/schema/parse_ddl)."""

import os

import pytest

from src.schema.parse_ddl import extract_schema_columns, classify_column


def test_extract_schema_columns():
    """Test standard SQL schema parsing."""
    schema_path = os.path.join("data", "sample_schema.sql")
    if not os.path.exists(schema_path):
        pytest.skip("sample_schema.sql missing")

    cols = extract_schema_columns(schema_path)
    tables = {c["table"] for c in cols}
    assert "usuarios" in tables
    assert "fichas_clinicas" in tables
    rut_cols = [c for c in cols if c["column"] == "rut"]
    assert len(rut_cols) >= 1


def test_classify_column_rut():
    """Identifiers should be flagged as Alto."""
    res = classify_column("rut", "usuarios")
    assert res["riesgo"] == "Alto"
    assert "identificador" in res["categoria"].lower()


def test_classify_column_email():
    """Contact info should be flagged as Medio."""
    res = classify_column("email", "usuarios")
    assert res["riesgo"] == "Medio"
    assert "contacto" in res["categoria"].lower()


def test_classify_column_salud():
    """Health data should be flagged as Alto / Sensibles."""
    res = classify_column("diagnostico_medico", "fichas_clinicas")
    assert res["riesgo"] == "Alto"
    assert "sensible" in res["categoria"].lower() or "salud" in res["categoria"].lower()


def test_classify_column_unknown():
    """Unrecognized columns should return Desconocida."""
    res = classify_column("obs_interna", "configuracion")
    assert res["categoria"] == "Desconocida"
    assert res.get("unknown") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
