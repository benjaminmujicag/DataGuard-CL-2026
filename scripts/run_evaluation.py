#!/usr/bin/env python3
"""CLI para evaluación RAG (grid / quick sample) de CL-DataGuard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.golden_dataset import GOLDEN_DATASET  # noqa: E402
from src.evaluation.rag_evaluator import print_results_only, RAGEvaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="CL-DataGuard RAG hyperparameter evaluation")
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick: 20 combinaciones aleatorias; full: 108 combinaciones",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Mostrar último CSV / best_config sin evaluar",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="No pedir confirmación interactiva (útil para corridas largas)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignorar checkpoint y empezar de cero",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para el muestreo en modo quick (si no se pasa: aleatoria por corrida; env RAG_EVAL_SEED para fijar)",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Limitar número de combinaciones (prueba humo; p. ej. 1)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        dest="corpus_dir",
        help=(
            "Carpeta con los 2 .md del corpus (ley_mixta.md, ley_21719.md); "
            "por defecto LEGAL_CORPUS_DIR o PDF_DIR o ./data/leyes_base"
        ),
    )
    args = parser.parse_args()

    if args.results_only:
        return print_results_only()

    n_golden = len(GOLDEN_DATASET)
    planned = 20 if args.mode == "quick" else 144
    if args.max_combos is not None:
        planned = min(planned, max(0, args.max_combos))

    print("🔍 CL-DataGuard RAG Evaluator")
    if args.mode == "quick":
        print("Modo: quick (muestra aleatoria de 20 del grid total de 144)")
    else:
        print("Modo: full (144 combinaciones)")
    if args.max_combos is not None:
        print(f"Límite activo: --max-combos={args.max_combos} → se evaluarán hasta {planned} combinaciones")
    print(f"Combinaciones a ejecutar en esta corrida: {planned}")
    print(f"Golden dataset: {n_golden} preguntas")
    print("Modelos: llama3.1 + nomic-embed-text (Ollama local)")
    print("Resultados en: outputs/rag_eval/")
    print("⚠️  Verifica que Ollama está corriendo antes de continuar.")
    if not args.yes:
        ans = input("¿Continuar? [s/N]: ").strip().lower()
        if ans not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado.")
            return 1

    ev = RAGEvaluator(
        mode=args.mode,
        seed=args.seed,
        pdf_dir=args.corpus_dir,
        fresh=args.fresh,
    )
    ev.run(max_combinations=args.max_combos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
