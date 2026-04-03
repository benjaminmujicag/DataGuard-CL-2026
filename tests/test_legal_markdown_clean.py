"""Tests for legal Markdown cleaning heuristics."""

from src.ingestion.legal_markdown_clean import (
    body_to_continuous_for_chunking,
    clean_legal_markdown_body,
    clean_legal_markdown_document,
    fidelity_fingerprint,
    split_frontmatter,
    verify_chunking_fidelity,
)


def test_split_frontmatter() -> None:
    text = "---\na: 1\n---\n\n# T\n\nbody\n"
    fm, body = split_frontmatter(text)
    assert fm is not None
    assert "a: 1" in fm
    assert body.startswith("# T")


def test_removes_bcn_boilerplate_and_margin_column() -> None:
    raw = """# Ley mixta

## Página 1

Ley 19628
Biblioteca del Congreso Nacional de Chile - www.leychile.cl - documento generado el 02-Abr-2026
página 1 de 57
SOBRE PROTECCION DE LOS DATOS PERSONALES                        Ley 21719
                                                                Art. primero N° 1)
     Teniendo presente que el H. Congreso Nacional ha dado      D.O. 13.12.2024
su aprobación al siguiente
"""
    out = clean_legal_markdown_body(raw)
    assert "Biblioteca del Congreso" not in out
    assert "página 1 de 57" not in out
    assert "Ley 21719" not in out
    assert "Art. primero" not in out
    assert "D.O. 13.12.2024" not in out
    assert "Teniendo presente" in out
    assert "su aprobación al siguiente" in out


def test_continuous_chunking_preserves_fidelity() -> None:
    raw = """# Ley X
## Página 1
Artículo 1°.- Uno.
## Página 2
Artículo 2°.- Dos.
"""
    paged = clean_legal_markdown_body(raw)
    cont = body_to_continuous_for_chunking(paged)
    assert "## Página" not in cont
    assert fidelity_fingerprint(paged) == fidelity_fingerprint(cont)
    ok, rep = verify_chunking_fidelity(paged, cont)
    assert ok is True
    assert rep["articulo_count_fingerprint"] == 2


def test_frontmatter_preserved_and_tagged() -> None:
    doc = """---
source_pdf: x.pdf
---
# Title

Biblioteca del Congreso Nacional de Chile - x
Hola mundo
"""
    cleaned = clean_legal_markdown_document(doc)
    assert "source_pdf: x.pdf" in cleaned
    assert "markdown_cleaner: legal_markdown_clean" in cleaned
    assert "Biblioteca del Congreso" not in cleaned
    assert "Hola mundo" in cleaned
