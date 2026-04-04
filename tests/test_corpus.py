"""Tests for the legal corpus loader — no Ollama required.

Validates that:
1. Both .md corpus files exist and have content.
2. load_legal_corpus() returns exactly 2 documents with the correct labels.
3. ley_mixta has priority 0 (appears first) over ley_21719 (priority 1).
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from src.ingestion.legal_corpus import load_legal_corpus, resolve_legal_md_paths

# The corpus directory relative to the project root.
CORPUS_DIR = str(Path(__file__).parent.parent / "data" / "leyes_base")

EXPECTED_LABELS = {
    "Ley mixta / transición",
    "Ley 21.719",
}

EXPECTED_FIRST_LABEL = "Ley mixta / transición"  # ley_mixta must be priority 0


@pytest.mark.parametrize("label", sorted(EXPECTED_LABELS))
def test_corpus_files_exist_and_have_content(label: str) -> None:
    """Both corpus .md files must exist and contain at least 1 KB of text."""
    resolved = resolve_legal_md_paths(CORPUS_DIR)
    labels_found = {lbl for _, lbl, _ in resolved}
    assert label in labels_found, (
        f"Etiqueta '{label}' no encontrada. Encontradas: {labels_found}. "
        "Verifica que los .md estén en data/leyes_base/."
    )
    for path, lbl, _ in resolved:
        if lbl == label:
            assert path.exists(), f"Archivo no encontrado: {path}"
            assert path.stat().st_size > 1024, (
                f"Archivo demasiado pequeño ({path.stat().st_size} bytes): {path}. "
                "Puede estar vacío o corrupto."
            )


def test_corpus_returns_two_documents() -> None:
    """load_legal_corpus() must return exactly 2 LangChain Documents."""
    docs = load_legal_corpus(CORPUS_DIR)
    assert len(docs) == 2, (
        f"Se esperaban 2 documentos, se obtuvieron {len(docs)}. "
        "Verifica que solo existan ley_mixta.md y ley_21719.md en data/leyes_base/."
    )
    labels_in_docs = {d.metadata.get("legal_corpus_label", "") for d in docs}
    assert labels_in_docs == EXPECTED_LABELS, (
        f"Labels incorrectos. Esperados: {EXPECTED_LABELS}. Encontrados: {labels_in_docs}"
    )


def test_corpus_priority_order() -> None:
    """ley_mixta.md (priority 0) must appear before ley_21719.md (priority 1)."""
    resolved = resolve_legal_md_paths(CORPUS_DIR)
    assert len(resolved) >= 2, "Se esperaban al menos 2 archivos en el corpus."
    first_label = resolved[0][1]
    assert first_label == EXPECTED_FIRST_LABEL, (
        f"La primera fuente debe ser '{EXPECTED_FIRST_LABEL}' (prioridad 0), "
        f"pero se encontró '{first_label}'. "
        "Verifica el orden en legal_corpus.py."
    )
