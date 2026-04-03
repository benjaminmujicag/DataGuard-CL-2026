"""CLI: export legal PDF corpus to Markdown for review and cleanup.

Loads ``PDF_DIR`` from the environment (see ``.env`` / ``.env.example``).

Modes:
  - ``--mode slots`` (default): same three PDFs as ingesta; falla si falta alguno.
  - ``--mode all``: todos los ``.pdf`` en la carpeta (útil si aún no tienes la 19.628).

Salida por defecto: ``./outputs/leyes_md`` (no commiteada por ``*.md`` en ``.gitignore``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo root = parent of scripts/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.pdf_to_markdown import (  # noqa: E402
    convert_all_pdfs_in_dir_to_markdown,
    convert_corpus_slots_to_markdown,
    summarize_markdown_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF del corpus legal → Markdown")
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Carpeta con PDFs (default: env PDF_DIR o ./data/leyes_base)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Carpeta de salida .md (default: env LEGAL_MD_OUTPUT_DIR o ./outputs/leyes_md)",
    )
    parser.add_argument(
        "--mode",
        choices=("slots", "all"),
        default="slots",
        help="slots = los 3 PDFs obligatorios del corpus; all = cualquier .pdf en la carpeta",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe .md existentes",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Imprime estadísticas y muestra de cada archivo generado",
    )
    args = parser.parse_args()

    pdf_dir = args.pdf_dir or os.getenv("PDF_DIR", "./data/leyes_base")
    out_dir = args.out_dir or os.getenv("LEGAL_MD_OUTPUT_DIR", "./outputs/leyes_md")

    pdf_path = Path(pdf_dir).expanduser()
    if not pdf_path.is_dir():
        print(f"ERROR: no existe el directorio de PDFs: {pdf_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.mode == "slots":
            paths = convert_corpus_slots_to_markdown(
                pdf_path, out_dir, overwrite=args.overwrite
            )
        else:
            paths = convert_all_pdfs_in_dir_to_markdown(
                pdf_path, out_dir, overwrite=args.overwrite
            )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Sugerencia: usa --mode all para convertir solo los PDF que tengas en la carpeta.",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)

    out_resolved = Path(out_dir).expanduser().resolve()
    print(f"Listo. Archivos en: {out_resolved}")
    for p in paths:
        rel = p.relative_to(out_resolved) if p.is_relative_to(out_resolved) else p
        print(f"  · {rel}")

    if args.summary:
        print("\n--- Resumen ---")
        for p in sorted(set(paths)):
            if p.suffix.lower() == ".md" and p.is_file():
                s = summarize_markdown_file(p)
                print(f"\n{s['path']}")
                print(f"  secciones '## Página': {s['page_sections']}")
                print(f"  caracteres (cuerpo): {s['body_chars']}")
                print(f"  muestra: {s['sample']!r}")


if __name__ == "__main__":
    main()
