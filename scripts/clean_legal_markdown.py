"""Clean exported legal Markdown (BCN boilerplate + right-margin amendment refs).

Reads ``*.md`` from ``--input-dir`` (default: env LEGAL_MD_DIR or ./outputs/leyes_md),
writes same filenames to ``--output-dir``.

Without ``--chunking``: paged output (keeps ``## Página N``), default dir ``./outputs/leyes_md_clean``.

With ``--chunking``: one continuous flow for text splitters (drops ``## Página N`` only),
default dir ``./outputs/leyes_md_chunking``. Use ``--verify`` to assert fidelity vs paged clean.

Example:
  python scripts/clean_legal_markdown.py
  python scripts/clean_legal_markdown.py --chunking --verify
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.legal_markdown_clean import (  # noqa: E402
    clean_markdown_file,
    clean_markdown_file_for_chunking,
    verify_chunking_fidelity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpia .md legales (ruido BCN + columna derecha)")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Carpeta con .md sucios (default: LEGAL_MD_DIR o ./outputs/leyes_md)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Salida (default: LEGAL_MD_CHUNKING_DIR si --chunking, si no LEGAL_MD_CLEAN_DIR)",
    )
    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Salida continua para chunking (sin ## Página N)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Comprueba huella y conteo Artículo vs cuerpo paginado (solo con --chunking)",
    )
    args = parser.parse_args()

    in_dir = Path(
        args.input_dir
        or os.getenv("LEGAL_MD_DIR", "./outputs/leyes_md")
    ).expanduser().resolve()

    default_out = (
        os.getenv("LEGAL_MD_CHUNKING_DIR", "./outputs/leyes_md_chunking")
        if args.chunking
        else os.getenv("LEGAL_MD_CLEAN_DIR", "./outputs/leyes_md_clean")
    )
    out_dir = Path(args.output_dir or default_out).expanduser().resolve()

    if not in_dir.is_dir():
        print(f"ERROR: no existe input-dir: {in_dir}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(in_dir.glob("*.md"))
    if not md_files:
        print(f"ERROR: no hay .md en {in_dir}", file=sys.stderr)
        sys.exit(2)

    all_ok = True
    for src in md_files:
        dest = out_dir / src.name
        if args.chunking:
            paged, continuous = clean_markdown_file_for_chunking(src, dest)
            print(f"OK  {src.name} -> {dest} (continuous)")
            if args.verify:
                ok, rep = verify_chunking_fidelity(paged, continuous)
                print(
                    f"    verify: fingerprint_ok={rep['fingerprint_match']} "
                    f"articulos_fp={rep['articulo_count_fingerprint']} "
                    f"(raw p/c {rep['articulo_count_paged_raw']}/{rep['articulo_count_continuous_raw']}) "
                    f"fp_len={rep['fingerprint_len_chars']}"
                )
                if not ok:
                    all_ok = False
                    print(f"    FALLO verificación en {src.name}: {rep}", file=sys.stderr)
        else:
            clean_markdown_file(src, dest)
            print(f"OK  {src.name} -> {dest}")

    print(f"\nListo: {len(md_files)} archivos en {out_dir}")
    if args.chunking and args.verify and not all_ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
