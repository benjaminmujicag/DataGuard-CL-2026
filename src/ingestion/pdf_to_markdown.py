"""Extract legal PDFs to Markdown files for inspection and manual or scripted cleanup.

Uses PyMuPDF for text extraction (often cleaner layout than raw PyPDF for some documents).
Output files are plain Markdown with YAML frontmatter (source path, legal label, rank when known).

# Review: opus-4.6 · 2026-04-03
# Auxiliary conversion tool (PDF -> .md). Updated to inline resolve_legal_pdf_paths
# since the main corpus module now resolves .md only. PyMuPDF extraction is correct. Approved.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.ingestion.legal_corpus import LEGAL_CORPUS_SLOTS

# PDF-specific slot names for the conversion tool (maps .md slot names back to .pdf).
_PDF_SLOT_NAMES: list[tuple[int, str, tuple[str, ...]]] = [
    (0, "Ley 21.719", ("ley_21719.pdf",)),
    (1, "Ley mixta / transición", ("ley_mixta_sucia.pdf",)),
    (2, "Ley 19.628", ("ley_19628.pdf", "ley_19.628.pdf", "ley_19_628.pdf", "ley19628.pdf")),
]


def resolve_legal_pdf_paths(pdf_dir: str | Path) -> list[tuple[Path, str, int]]:
    """Resolve three PDF files in ``pdf_dir`` for conversion to Markdown.

    Args:
        pdf_dir: Directory containing the source PDFs.

    Returns:
        List of ``(absolute_path, legal_label, priority_rank)``.

    Raises:
        FileNotFoundError: If any required slot is missing.
    """
    root = Path(pdf_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {root}")
    index: dict[str, Path] = {}
    for f in root.iterdir():
        if f.is_file() and f.suffix.lower() == ".pdf":
            index[f.name.lower()] = f.resolve()
    resolved: list[tuple[Path, str, int]] = []
    missing: list[str] = []
    for rank, label, candidates in _PDF_SLOT_NAMES:
        hit = next((index[c.lower()] for c in candidates if c.lower() in index), None)
        if hit is None:
            missing.append(f"  - {label}: se buscó uno de {', '.join(candidates)}")
        else:
            resolved.append((hit, label, rank))
    if missing:
        found = ", ".join(sorted(p.name for p in index.values())) or "(ninguno)"
        raise FileNotFoundError(
            f"Faltan PDFs en {root.resolve()}:\n" + "\n".join(missing) + f"\nPresentes: {found}"
        )
    return resolved


def _escape_md_line_starts(text: str) -> str:
    """Prefix lines that would be interpreted as Markdown headings or lists.

    Args:
        text: Raw page text.

    Returns:
        Text safe to paste under a ``## Página N`` block without changing structure.
    """
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out.append("\\" + line if line.startswith("#") else line[: len(line) - len(stripped)] + "\\" + stripped)
        elif re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}\\{stripped}")
        else:
            out.append(line)
    return "\n".join(out)


def extract_pdf_text_by_pages(pdf_path: Path) -> list[str]:
    """Extract plain text per page with PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of page texts (0-based index matches page number in PDF).

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If the document cannot be opened.
    """
    path = pdf_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF no encontrado: {path}")
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir PDF {path}: {e}") from e
    pages: list[str] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(text.strip())
    finally:
        doc.close()
    return pages


def _yaml_escape(value: str) -> str:
    if not value:
        return '""'
    if any(c in value for c in '":\n#'):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


def build_markdown_for_law(
    pdf_path: Path,
    *,
    legal_label: str | None = None,
    legal_priority_rank: int | None = None,
    escape_line_starts: bool = True,
) -> str:
    """Build one Markdown document string from a PDF path and optional corpus metadata.

    Args:
        pdf_path: Source PDF.
        legal_label: Human label (e.g. Ley 21.719) when using corpus slots.
        legal_priority_rank: RAG priority rank when known.
        escape_line_starts: If True, escape lines that would parse as MD headings/lists.

    Returns:
        Full file content (frontmatter + body).
    """
    page_texts = extract_pdf_text_by_pages(pdf_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_lines = [
        "---",
        f"source_pdf: {_yaml_escape(pdf_path.name)}",
        f"source_path: {_yaml_escape(str(pdf_path.resolve()))}",
        f"extracted_at_utc: {now}",
        "extractor: pymupdf",
    ]
    if legal_label is not None:
        meta_lines.append(f"legal_corpus_label: {_yaml_escape(legal_label)}")
    if legal_priority_rank is not None:
        meta_lines.append(f"legal_priority_rank: {legal_priority_rank}")
    meta_lines.append(f"page_count: {len(page_texts)}")
    meta_lines.append("---")
    meta_lines.append("")

    title = legal_label or pdf_path.stem.replace("_", " ")
    body_lines: list[str] = [f"# {title}", ""]
    for i, raw in enumerate(page_texts, start=1):
        body_lines.append(f"## Página {i}")
        body_lines.append("")
        chunk = raw if not escape_line_starts else _escape_md_line_starts(raw)
        body_lines.append(chunk if chunk else "_[página sin texto extraíble]_")
        body_lines.append("")

    return "\n".join(meta_lines + body_lines).rstrip() + "\n"


def convert_corpus_slots_to_markdown(
    pdf_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Convert the three legal corpus PDFs (same resolution as ingest) to Markdown files.

    Args:
        pdf_dir: Directory containing ``ley_21719.pdf``, etc.
        output_dir: Directory to write ``.md`` files (created if missing).
        overwrite: If False, skip existing files.

    Returns:
        Paths of written or skipped files.

    Raises:
        FileNotFoundError: If corpus PDFs are incomplete (same as ``resolve_legal_pdf_paths``).
    """
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    resolved = resolve_legal_pdf_paths(pdf_dir)
    written: list[Path] = []
    for path, label, rank in resolved:
        md_name = path.stem + ".md"
        dest = out_root / md_name
        if dest.exists() and not overwrite:
            written.append(dest)
            continue
        content = build_markdown_for_law(
            path, legal_label=label, legal_priority_rank=rank, escape_line_starts=True
        )
        dest.write_text(content, encoding="utf-8")
        written.append(dest)
    return written


def _pdf_dir_index(pdf_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not pdf_dir.is_dir():
        return out
    for f in pdf_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".pdf":
            out[f.name.lower()] = f.resolve()
    return out


def _label_for_basename(basename: str) -> tuple[str | None, int | None]:
    lower = basename.lower()
    for rank, label, candidates in LEGAL_CORPUS_SLOTS:
        for c in candidates:
            if c.lower() == lower:
                return label, rank
    return None, None


def convert_all_pdfs_in_dir_to_markdown(
    pdf_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every ``.pdf`` in ``pdf_dir`` to Markdown (no requirement for full corpus).

    Filenames are sorted. Metadata ``legal_corpus_label`` / ``legal_priority_rank`` are set when
    the basename matches a known corpus slot.

    Args:
        pdf_dir: Folder to scan.
        output_dir: Output folder for ``.md`` files.
        overwrite: Overwrite existing Markdown files.

    Returns:
        Paths of written or skipped files.
    """
    root = Path(pdf_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No es directorio: {root}")
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    index = _pdf_dir_index(root)
    written: list[Path] = []
    for name in sorted(index.keys()):
        path = index[name]
        label, rank = _label_for_basename(path.name)
        md_name = path.stem + ".md"
        dest = out_root / md_name
        if dest.exists() and not overwrite:
            written.append(dest)
            continue
        content = build_markdown_for_law(
            path,
            legal_label=label,
            legal_priority_rank=rank,
            escape_line_starts=True,
        )
        dest.write_text(content, encoding="utf-8")
        written.append(dest)
    return written


def summarize_markdown_file(md_path: Path, *, sample_chars: int = 400) -> dict[str, Any]:
    """Lightweight stats for analyzing extraction quality (CLI / notebooks).

    Args:
        md_path: Path to a generated ``.md`` file.
        sample_chars: Length of text sample after frontmatter.

    Returns:
        Dict with keys: path, page_sections, body_chars, sample.
    """
    raw = md_path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        end = raw.find("\n---\n", 3)
        body = raw[end + 5 :] if end != -1 else raw
    else:
        body = raw
    page_sections = body.count("\n## Página ")
    sample = body[:sample_chars].replace("\n", " ")
    return {
        "path": str(md_path),
        "page_sections": page_sections,
        "body_chars": len(body),
        "sample": sample + ("…" if len(body) > sample_chars else ""),
    }
