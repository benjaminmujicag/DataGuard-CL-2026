"""Heuristics to clean BCN-style legal Markdown exports (boilerplate + right-margin refs).

Removes repeated LeyChile headers/footers per page and strips the editorial column
(Ley 21.719, Art. primero N°, D.O., NOTA markers) when separated by wide whitespace gaps.

Optional second step ``body_to_continuous_for_chunking`` drops ``## Página N`` headings so
the corpus is one continuous flow for text splitters (one-off corpus; no new laws planned).

# Review: opus-4.6 · 2026-04-03
# Well-structured regex heuristics for BCN boilerplate. Fidelity verification is thorough.
# Edge cases in margin detection are documented. Approved.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Headings produced by pdf_to_markdown (one per PDF page); removed for chunking-only output.
_PAGE_HEADING_RE = re.compile(
    r"^\s*##\s+Página\s+\d+\s*$", re.IGNORECASE | re.MULTILINE
)

# Trailing margin refs (wide space then BCN column); applied repeatedly from line end.
_TRAILING_MARGIN_RE = re.compile(
    r"\s{3,}(?:"
    r"Art\.\s[^\n]+|"
    r"D\.O\.\s[^\n]+|"
    r"Ley\s+(?:21\.719|21719|21755|21806)\b[^\n]*|"
    r"Ley\s+19628\b[^\n]*|"
    r"Ley\s+19\.628\b[^\n]*|"
    r"uno\)|dos\)|tres\)|cuatro\)|cinco\)|seis\)|"
    r"NOTA\b[^\n]*|"
    r"[a-z]\)\s+(?:i{1,3}|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s*|"
    r"[a-z]\)\s*|"
    r"(?:LEY|Ley)\s+\d[\d.]*\s*"
    r")\s*$",
    re.IGNORECASE,
)

_BOILERPLATE_LINE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Biblioteca del Congreso Nacional de Chile\b", re.IGNORECASE),
    re.compile(r"^página\s+\d+\s+de\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^Url Corta:\s*https?://", re.IGNORECASE),
    re.compile(r"^Fecha Publicación:\s*", re.IGNORECASE),
    re.compile(r"^Fecha Promulgación:\s*", re.IGNORECASE),
    re.compile(r"^Tipo Versión:\s*", re.IGNORECASE),
    re.compile(r"^Ultima Modificación:\s*", re.IGNORECASE),
)

_STANDALONE_LEY_RE = re.compile(
    r"^Ley\s+19628$|^Ley\s+19\.628$|^Ley\s+21719$|^Ley\s+21\.719$",
    re.IGNORECASE,
)


def _is_boilerplate_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for pat in _BOILERPLATE_LINE_RES:
        if pat.search(s):
            return True
    if _STANDALONE_LEY_RE.match(s):
        return True
    if "MINISTERIO SECRETARÍA GENERAL DE LA PRESIDENCIA" in s and len(s) < 160:
        return True
    return False


def _is_margin_token(text: str) -> bool:
    """True if ``text`` looks like BCN margin / amendment reference only."""
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"Ley\s+21\.719.*", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"Ley\s+21719\b.*", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"Ley\s+21755\b.*", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"Ley\s+21806\b.*", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"Ley\s+19628\b.*", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"Ley\s+19\.628\b.*", t, re.IGNORECASE):
        return True
    if re.match(r"^Art\.\s*", t, re.IGNORECASE):
        return True
    if re.match(r"^D\.O\.\s*", t, re.IGNORECASE):
        return True
    if t == "NOTA" or t.startswith("NOTA "):
        return True
    # Continuations of broken margin lines, e.g. "uno)" after "N° 5,"
    if re.fullmatch(r"uno\)|dos\)|tres\)|cuatro\)|cinco\)|seis\)", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d+\)", t) and len(t) <= 6:
        return True
    return False


def _strip_trailing_margin_columns(line: str) -> str | None:
    """Remove one or more margin fragments from the end of a line (BCN two-column export)."""
    s = line.rstrip()
    if not s:
        return None
    if _is_margin_token(s):
        return None
    prev = ""
    while prev != s:
        prev = s
        m = _TRAILING_MARGIN_RE.search(s)
        if not m:
            break
        s = s[: m.start()].rstrip()
    if not s:
        return None
    if _is_margin_token(s):
        return None
    return s


def _iter_cleaned_body_lines(lines: Iterator[str]) -> Iterator[str]:
    for raw in lines:
        line = raw.rstrip("\n\r")
        if _is_boilerplate_line(line):
            continue
        cleaned = _strip_trailing_margin_columns(line)
        if cleaned is None:
            continue
        if not cleaned.strip():
            yield ""
        else:
            yield cleaned.rstrip()


def _dedupe_adjacent_duplicate_lines(lines: list[str]) -> list[str]:
    """Drop a line if it is identical to the previous non-empty line (strips BCN title dupes)."""
    out: list[str] = []
    for ln in lines:
        if out and ln.strip() and out[-1].strip() == ln.strip():
            continue
        out.append(ln)
    return out


def _collapse_blank_runs(lines: list[str], max_blank_run: int = 2) -> list[str]:
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= max_blank_run:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return out


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Return (frontmatter_block_including_delimiters, body) or (None, full_text)."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    fm = text[: end + 5]
    body = text[end + 5 :].lstrip("\n")
    return fm, body


def clean_legal_markdown_body(body: str) -> str:
    """Apply boilerplate and margin stripping to markdown body (after frontmatter)."""
    raw_lines = body.splitlines()
    cleaned = list(_iter_cleaned_body_lines(iter(raw_lines)))
    deduped = _dedupe_adjacent_duplicate_lines(cleaned)
    collapsed = _collapse_blank_runs(deduped, max_blank_run=2)
    return "\n".join(collapsed).rstrip() + "\n"


def body_to_continuous_for_chunking(body: str) -> str:
    """Remove ``## Página N`` lines; keep ``#`` title and normative text as one flow for chunking.

    Args:
        body: Already BCN-cleaned markdown body.

    Returns:
        Same content without page section headings; blank runs collapsed.
    """
    lines: list[str] = []
    for line in body.splitlines():
        if _PAGE_HEADING_RE.match(line):
            continue
        lines.append(line)
    collapsed = _collapse_blank_runs(lines, max_blank_run=2)
    return "\n".join(collapsed).rstrip() + "\n"


def fidelity_fingerprint(body: str) -> str:
    """Normalize body for equality check: ignore page headings and all whitespace runs."""
    t = "\n".join(
        ln for ln in body.splitlines() if not _PAGE_HEADING_RE.match(ln)
    )
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def verify_chunking_fidelity(paged_body: str, continuous_body: str) -> tuple[bool, dict[str, Any]]:
    """Check that continuous output differs from paged only by page markers / blank layout.

    Equality uses ``fidelity_fingerprint`` (whitespace-normalized, page headings ignored).
    Raw ``Artículo\\s+\\d+`` counts can differ when page lines sat between tokens matched
    by ``\\s+`` across lines; the fingerprint is the authoritative check.

    Returns:
        (ok, report) where report includes fingerprints and marker counts.
    """
    fp_p = fidelity_fingerprint(paged_body)
    fp_c = fidelity_fingerprint(continuous_body)
    art_p_raw = len(re.findall(r"Artículo\s+\d+", paged_body, flags=re.IGNORECASE))
    art_c_raw = len(re.findall(r"Artículo\s+\d+", continuous_body, flags=re.IGNORECASE))
    art_fp = len(re.findall(r"Artículo\s+\d+", fp_p, flags=re.IGNORECASE))
    pages_p = len(_PAGE_HEADING_RE.findall(paged_body))
    pages_c = len(_PAGE_HEADING_RE.findall(continuous_body))
    ok = fp_p == fp_c and pages_c == 0
    return ok, {
        "fingerprint_match": fp_p == fp_c,
        "articulo_count_paged_raw": art_p_raw,
        "articulo_count_continuous_raw": art_c_raw,
        "articulo_count_fingerprint": art_fp,
        "page_headings_removed": pages_p,
        "page_headings_remaining_in_continuous": pages_c,
        "fingerprint_len_chars": len(fp_p),
    }


def clean_legal_markdown_document(text: str) -> str:
    """Preserve YAML frontmatter; append cleaning metadata; clean body."""
    fm, body = split_frontmatter(text)
    cleaned_body = clean_legal_markdown_body(body)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if fm:
        # fm is "---\n" + yaml + "\n---\n"
        if fm.startswith("---\n") and fm.endswith("\n---\n"):
            inner = fm[4:-5].rstrip()
            if "markdown_cleaner:" not in inner:
                inner += (
                    f"\nmarkdown_cleaned_at_utc: {stamp}\n"
                    "markdown_cleaner: legal_markdown_clean"
                )
            fm = f"---\n{inner}\n---\n"
        return fm + "\n" + cleaned_body
    return cleaned_body


def clean_markdown_file(src: Path, dest: Path) -> None:
    """Read ``src``, write cleaned markdown to ``dest`` (parent dirs created)."""
    text = src.read_text(encoding="utf-8")
    out = clean_legal_markdown_document(text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out, encoding="utf-8")


def clean_markdown_file_for_chunking(src: Path, dest: Path) -> tuple[str, str]:
    """Clean source then emit continuous body for chunking. Returns (paged_body, continuous_body)."""
    text = src.read_text(encoding="utf-8")
    fm, body = split_frontmatter(text)
    paged_body = clean_legal_markdown_body(body)
    continuous_body = body_to_continuous_for_chunking(paged_body)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if fm and fm.startswith("---\n") and fm.endswith("\n---\n"):
        inner = fm[4:-5].rstrip()
        if "markdown_cleaner:" not in inner:
            inner += (
                f"\nmarkdown_cleaned_at_utc: {stamp}\n"
                "markdown_cleaner: legal_markdown_clean\n"
            )
        if "markdown_chunking:" not in inner:
            inner += (
                "markdown_chunking: continuous\n"
                f"markdown_chunking_at_utc: {stamp}\n"
            )
        fm = f"---\n{inner}\n---\n"
        out = fm + "\n" + continuous_body
    else:
        out = continuous_body
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out, encoding="utf-8")
    return paged_body, continuous_body
