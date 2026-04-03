"""Corpus legal fijo: tres Markdowns en ``LEGAL_CORPUS_DIR`` con prioridad explícita para RAG.

Orden de autoridad en recuperación y en el prompt (menor ``legal_priority_rank`` = mayor prioridad):

1. Ley 21.719 — ``ley_21719.md``
2. Ley mixta / transición — ``ley_mixta_sucia.md``
3. Ley 19.628 — ``ley_19628.md`` (y alias de nombre aceptados)

Solo se indexan estos archivos (no todo el directorio), para evitar ruido de archivos extra.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

LEGAL_CORPUS_SLOTS: list[tuple[int, str, tuple[str, ...]]] = [
    (0, "Ley 21.719", ("ley_21719.md",)),
    (1, "Ley mixta / transición", ("ley_mixta_sucia.md",)),
    (
        2,
        "Ley 19.628",
        (
            "ley_19628.md",
            "ley_19.628.md",
            "ley_19_628.md",
            "ley19628.md",
        ),
    ),
]


def _md_dir_index(md_dir: Path) -> dict[str, Path]:
    """Map lowercase basename -> resolved path for all .md files in directory."""
    out: dict[str, Path] = {}
    if not md_dir.is_dir():
        return out
    for f in md_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".md":
            out[f.name.lower()] = f.resolve()
    return out


def _strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (``---`` delimited block at start) if present."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---\n", 3)
    if end == -1:
        return text
    return text[end + 5:].lstrip("\n")


def resolve_legal_md_paths(md_dir: str | Path) -> list[tuple[Path, str, int]]:
    """Resolve the three mandatory Markdown files in ``md_dir`` respecting priority.

    Args:
        md_dir: Directory containing the ``.md`` files (e.g. ``./data/leyes_base``).

    Returns:
        List of ``(absolute_path, legal_label, priority_rank)`` in loading order.

    Raises:
        FileNotFoundError: If any required slot is missing or the directory does not exist.
    """
    root = Path(md_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Directorio de corpus legal no encontrado o no es carpeta: {root}"
        )
    index = _md_dir_index(root)
    resolved: list[tuple[Path, str, int]] = []
    missing_lines: list[str] = []

    for rank, label, candidates in LEGAL_CORPUS_SLOTS:
        hit: Path | None = None
        for name in candidates:
            key = name.lower()
            if key in index:
                hit = index[key]
                break
        if hit is None:
            missing_lines.append(
                f"  - {label}: se buscó uno de {', '.join(candidates)}"
            )
        else:
            resolved.append((hit, label, rank))

    if missing_lines:
        found = ", ".join(sorted(p.name for p in index.values())) or "(ninguno)"
        msg = (
            f"Faltan archivos .md obligatorios del corpus en {root.resolve()}:\n"
            + "\n".join(missing_lines)
            + f"\nArchivos .md presentes: {found}"
        )
        raise FileNotFoundError(msg)

    return resolved


def load_legal_corpus(md_dir: str | Path | None = None) -> list[Document]:
    """Load the three Markdown files as LangChain Documents with priority metadata.

    Each file is loaded as a single Document (full text). The YAML frontmatter
    is stripped so that only legal content is indexed.

    Args:
        md_dir: Folder with the law ``.md`` files.  Falls back to env var
            ``LEGAL_CORPUS_DIR`` then ``./data/leyes_base``.

    Returns:
        Documents in priority order: 21.719 first, then mixta, then 19.628.
    """
    if md_dir is None:
        md_dir = os.getenv("LEGAL_CORPUS_DIR", os.getenv("PDF_DIR", "./data/leyes_base"))
    slots = resolve_legal_md_paths(md_dir)
    all_docs: list[Document] = []
    for path, label, rank in slots:
        raw = path.read_text(encoding="utf-8")
        body = _strip_yaml_frontmatter(raw)
        doc = Document(
            page_content=body,
            metadata={
                "source": str(path),
                "legal_corpus_label": label,
                "legal_priority_rank": rank,
            },
        )
        all_docs.append(doc)
    return all_docs


def resolve_legal_corpus_dir(explicit_dir: str | None = None) -> Path:
    """Resolve the corpus directory (evaluation may override with explicit path).

    Precedence: ``explicit_dir`` (if file, its parent), then
    ``RAG_EVAL_CORPUS_DIR``, then ``LEGAL_CORPUS_DIR``, then ``PDF_DIR``,
    then ``./data/leyes_base``.
    """
    if explicit_dir:
        p = Path(explicit_dir).expanduser()
        if p.is_file():
            return p.parent.resolve()
        if p.is_dir():
            return p.resolve()
        raise FileNotFoundError(f"Ruta de corpus inexistente: {explicit_dir}")

    for env_var in ("RAG_EVAL_CORPUS_DIR", "LEGAL_CORPUS_DIR", "PDF_DIR"):
        val = os.getenv(env_var)
        if val:
            pe = Path(val).expanduser()
            if pe.is_file():
                return pe.parent.resolve()
            if pe.is_dir():
                return pe.resolve()
            raise FileNotFoundError(f"{env_var} apunta a ruta inválida: {val}")

    default = Path("./data/leyes_base").expanduser()
    if not default.is_dir():
        raise FileNotFoundError(
            f"No existe directorio para el corpus legal: {default.resolve()}. "
            "Configura LEGAL_CORPUS_DIR o PDF_DIR."
        )
    return default.resolve()


def format_legal_docs_for_prompt(docs: Sequence[Any]) -> str:
    """Format retrieved fragments ordered by normative priority (21.719 first)."""
    sorted_docs = sorted(
        docs,
        key=lambda d: int((getattr(d, "metadata", None) or {}).get("legal_priority_rank", 999)),
    )
    lines: list[str] = []
    for i, doc in enumerate(sorted_docs):
        meta = getattr(doc, "metadata", None) or {}
        basename = os.path.basename(str(meta.get("source", "Ley Desconocida")))
        label = meta.get("legal_corpus_label", "")
        tag = f"{basename}" + (f" — {label}" if label else "")
        lines.append(f"[{i + 1}] (Fuente: {tag}) \n{doc.page_content}")
    return "\n\n".join(lines)
