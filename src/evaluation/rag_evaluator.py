"""Hyperparameter search for RAG using keyword coverage over a golden dataset.

Uses temporary Chroma collections and persist directories; does not modify production
``ley_privacidad_cl`` in ``CHROMA_PERSIST_DIR``.

# Review: opus-4.6 · 2026-04-03
# Solid implementation: checkpoint/resume, timeout guards, CSV incremental logging,
# production-safe Chroma isolation. Updated import from resolve_legal_pdf_paths to
# resolve_legal_md_paths for .md corpus migration. Approved.
"""

from __future__ import annotations

import csv
import json
import os
import random
import secrets
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

from dotenv import load_dotenv

load_dotenv()
for _k, _v in (("ANONYMIZED_TELEMETRY", "False"), ("CHROMA_TELEMETRY", "false")):
    os.environ.setdefault(_k, _v)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.evaluation.golden_dataset import GOLDEN_DATASET
from src.ingestion.legal_corpus import (
    format_legal_docs_for_prompt,
    load_legal_corpus,
    resolve_legal_corpus_dir,
    resolve_legal_md_paths,
)

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

PARAM_GRID: dict[str, list[int | float]] = {
    "chunk_size": [500, 750, 1000, 1500],
    "chunk_overlap": [50, 100, 200],
    "top_k": [2, 3, 5],
    "temperature": [0.0, 0.1, 0.3],
}

DEFAULT_EVAL_CHROMA_ROOT = "./outputs/rag_eval/.chroma_scratch"
CHECKPOINT_NAME = "checkpoint.json"
INVOKE_TIMEOUT_SEC = float(os.getenv("RAG_EVAL_INVOKE_TIMEOUT_SEC", "180"))
INGEST_TIMEOUT_SEC = float(os.getenv("RAG_EVAL_INGEST_TIMEOUT_SEC", "600"))

# Mirrors production RAG prompt (retriever.py) for comparable evaluation.
_RAG_TEMPLATE = """Eres un experto y riguroso Asesor Legal Corporativo especializado en la normativa chilena, operando estrictamente bajo la Ley 21.719 que Regula la Protección y el Tratamiento de los Datos Personales y crea la Agencia de Protección de Datos Personales de Chile.

Tu tarea es responder a la pregunta del usuario utilizando de manera EXCLUSIVA el conocimiento jurídico proporcionado en "Fragmentos Relevantes".

Reglas de Operación:
1. Responde SIEMPRE en español, de forma clara, consultiva y profesional.
2. Basate ÚNICAMENTE en el contenido de los "Fragmentos Relevantes" para fundamentar tu respuesta.
3. Tienes permitido inferir, deducir y sintetizar conceptos si los fragmentos te dan suficiente contexto tácito (por ejemplo, analizar qué es un dato personal en base a cómo la ley norma su uso).
4. Si los fragmentos de plano no tienen NINGUNA relación con la pregunta o te es imposible deducir una respuesta con ellos, responde cortésmente que la ley no lo menciona explícitamente en el texto recuperado.
5. Cita siempre el Artículo al que haces referencia (y aclara si pertenece a la Ley 19.628 o a la Ley 21.719 repasando la etiqueta 'Fuente' del fragmento).
6. REGLA SUPREMA DE CONFLICTO: Si existe alguna discrepancia, choque de definiciones o actualización entre un fragmento etiquetado como fuente Ley 19.628 y un fragmento de la Ley 21.719, darás SIEMPRE PRIORIDAD ABSOLUTA a la información contenida en la Ley 21.719 por ser la ley modificatoria legalmente vigente.
7. PROHIBIDO inventar normativas externas a los fragmentos o alucinar artículos que no estén en el texto provisto.

---
Fragmentos Relevantes:
{context}

Pregunta del Usuario:
{question}
---

Respuesta Legal:"""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split documents into chunks (same strategy as ingestion)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def keyword_coverage_score(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in answer (case-insensitive)."""
    if not expected_keywords:
        return 0.0
    lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return hits / len(expected_keywords)


def _format_docs(docs: list[Any]) -> str:
    return format_legal_docs_for_prompt(docs)


def _run_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Any:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut: Future[Any] = pool.submit(fn)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            raise TimeoutError(f"Timeout after {timeout_sec}s") from None


def _build_rag_chain(
    persist_dir: str,
    collection_name: str,
    top_k: int,
    temperature: float,
    embedding_model: str,
    ollama_base_url: str,
    llm_model: str,
) -> Any:
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    llm = OllamaLLM(
        model=llm_model,
        base_url=ollama_base_url,
        temperature=temperature,
    )
    prompt = PromptTemplate.from_template(_RAG_TEMPLATE)
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def expand_param_grid() -> list[dict[str, int | float]]:
    """Cartesian product of PARAM_GRID."""
    keys = list(PARAM_GRID.keys())
    out: list[dict[str, int | float]] = []

    def rec(i: int, cur: dict[str, int | float]) -> None:
        if i == len(keys):
            out.append(dict(cur))
            return
        k = keys[i]
        for v in PARAM_GRID[k]:
            cur[k] = v
            rec(i + 1, cur)
        cur.pop(k, None)

    rec(0, {})
    return out


def sample_combinations(
    combinations: list[dict[str, int | float]], n: int, seed: int | None
) -> list[dict[str, int | float]]:
    if n >= len(combinations):
        return list(combinations)
    rng = random.Random(seed)
    return rng.sample(combinations, n)


def _combo_tuple(c: dict[str, int | float]) -> tuple[int, int, int, float]:
    return (
        int(c["chunk_size"]),
        int(c["chunk_overlap"]),
        int(c["top_k"]),
        float(c["temperature"]),
    )


def _combinations_match(
    a: list[dict[str, int | float]], b: list[dict[str, int | float]]
) -> bool:
    if len(a) != len(b):
        return False
    return all(_combo_tuple(x) == _combo_tuple(y) for x, y in zip(a, b))


def _resolve_eval_seed(explicit: int | None) -> int:
    """Semilla para muestreo quick: CLI > env RAG_EVAL_SEED > aleatoria por corrida.

    Sin semilla fija, cada ejecución ordena distinto el subconjunto de 20 combinaciones
    (la primera ya no es siempre la misma). Para repetir resultados: ``--seed`` o
    ``RAG_EVAL_SEED`` en ``.env``.
    """
    if explicit is not None:
        return explicit
    env = os.getenv("RAG_EVAL_SEED", "").strip()
    if env:
        return int(env)
    return secrets.randbelow(2**31 - 1)


class RAGEvaluator:
    """Grid search / random subsample evaluation with checkpointing and CSV logging."""

    def __init__(
        self,
        mode: str = "quick",
        seed: int | None = None,
        pdf_dir: str | None = None,
        output_dir: str | None = None,
        fresh: bool = False,
    ) -> None:
        self.mode = mode
        self.seed = _resolve_eval_seed(seed)
        self.pdf_dir = resolve_legal_corpus_dir(pdf_dir)
        root = _project_root()
        self.output_dir = Path(output_dir or (root / "outputs" / "rag_eval"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Chroma de evaluación: SIEMPRE fuera de CHROMA_PERSIST_DIR (producción).
        cr = Path(os.getenv("RAG_EVAL_CHROMA_ROOT", DEFAULT_EVAL_CHROMA_ROOT)).expanduser()
        self.chroma_root = (root / cr).resolve() if not cr.is_absolute() else cr.resolve()
        prod_chroma = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")).expanduser()
        prod_chroma = (
            (root / prod_chroma).resolve()
            if not prod_chroma.is_absolute()
            else prod_chroma.resolve()
        )
        if self.chroma_root == prod_chroma:
            raise ValueError(
                "RAG_EVAL_CHROMA_ROOT no puede ser el mismo directorio que "
                f"CHROMA_PERSIST_DIR ({prod_chroma}). La evaluación usa Chroma temporal "
                f"bajo outputs (por defecto {DEFAULT_EVAL_CHROMA_ROOT}) y no debe tocar "
                "la base de producción."
            )
        self.chroma_root.mkdir(parents=True, exist_ok=True)
        self.fresh = fresh

        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm_model = os.getenv("OLLAMA_MODEL", "llama3.1")

        self._run_timestamp: str | None = None
        self._csv_path: Path | None = None
        self._log_fp: TextIO | None = None
        self._checkpoint_path = self.output_dir / CHECKPOINT_NAME

    def _new_run_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _open_log(self, path: Path) -> TextIO:
        return open(path, "a", encoding="utf-8")

    def _log(self, line: str) -> None:
        if self._log_fp:
            self._log_fp.write(line + "\n")
            self._log_fp.flush()

    def _save_checkpoint(
        self,
        combinations: list[dict[str, int | float]],
        completed_ids: list[int],
        csv_path: Path,
        log_path: Path,
    ) -> None:
        payload = {
            "run_timestamp": self._run_timestamp,
            "mode": self.mode,
            "seed": self.seed,
            "combinations": combinations,
            "completed_combination_ids": completed_ids,
            "csv_path": str(csv_path),
            "log_path": str(log_path),
            "total_combinations": len(combinations),
        }
        self._checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_checkpoint(self) -> dict[str, Any] | None:
        if not self._checkpoint_path.is_file():
            return None
        try:
            return json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _ingest_combo(
        self,
        docs: list[Document],
        chunk_size: int,
        chunk_overlap: int,
        persist_dir: str,
        collection_name: str,
    ) -> None:
        def _do() -> None:
            chunks = split_documents(docs, chunk_size, chunk_overlap)
            embeddings = OllamaEmbeddings(
                model=self.embedding_model, base_url=self.ollama_base_url
            )
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )

        _run_with_timeout(_do, INGEST_TIMEOUT_SEC)

    def _cleanup_chroma_dir(self, persist_dir: Path) -> None:
        if persist_dir.is_dir():
            try:
                shutil.rmtree(persist_dir, ignore_errors=False)
            except OSError as e:
                self._log(f"WARN: no se pudo borrar {persist_dir}: {e}")

    def _evaluate_one_combo(
        self,
        combo: dict[str, int | float],
        combinacion_id: int,
        docs: list[Document],
    ) -> dict[str, Any]:
        chunk_size = int(combo["chunk_size"])
        chunk_overlap = int(combo["chunk_overlap"])
        top_k = int(combo["top_k"])
        temperature = float(combo["temperature"])

        collection_name = f"eval_{chunk_size}_{chunk_overlap}"
        persist_dir = str(
            self.chroma_root / f"run_{self._run_timestamp}" / f"combo_{combinacion_id}"
        )
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        row_ts = datetime.now().isoformat(timespec="seconds")
        failed = False

        try:
            self._ingest_combo(
                docs, chunk_size, chunk_overlap, persist_dir, collection_name
            )
        except Exception as e:
            self._log(f"FAIL ingest combo {combinacion_id}: {e}")
            failed = True

        if failed:
            self._cleanup_chroma_dir(Path(persist_dir))
            return {
                "timestamp": row_ts,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "temperature": temperature,
                "score_promedio": -1.0,
                "score_minimo": -1.0,
                "score_maximo": -1.0,
                "tiempo_promedio_segundos": -1.0,
                "mejor_pregunta": "",
                "peor_pregunta": "",
                "combinacion_id": combinacion_id,
            }

        chain = _build_rag_chain(
            persist_dir=persist_dir,
            collection_name=collection_name,
            top_k=top_k,
            temperature=temperature,
            embedding_model=self.embedding_model,
            ollama_base_url=self.ollama_base_url,
            llm_model=self.llm_model,
        )

        per_q_scores: list[tuple[float, str, float]] = []
        times: list[float] = []
        combo_failed = False
        n_q = len(GOLDEN_DATASET)

        self._log(
            f"combo {combinacion_id}: inicio evaluación golden ({n_q} preguntas, "
            f"chunk={chunk_size}/{chunk_overlap}, top_k={top_k}, temp={temperature})"
        )

        for qi, item in enumerate(GOLDEN_DATASET, start=1):
            question = str(item["question"])
            keywords = list(item["expected_keywords"])  # type: ignore[list-item]
            t0 = time.perf_counter()
            answer = ""
            q_status = "OK"
            err_detail = ""
            try:

                def _invoke() -> str:
                    return str(chain.invoke(question))

                answer = _run_with_timeout(_invoke, INVOKE_TIMEOUT_SEC)
            except TimeoutError as e:
                combo_failed = True
                q_status = "TIMEOUT"
                err_detail = str(e)[:120]
            except Exception as e:
                combo_failed = True
                q_status = "FAIL"
                err_detail = str(e)[:120]
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            sc = keyword_coverage_score(answer, keywords)
            per_q_scores.append((sc, question, elapsed))
            hits_kw = sum(1 for kw in keywords if kw.lower() in answer.lower())
            preview = question.replace("\n", " ").strip()[:100]
            tail = f" err={err_detail}" if err_detail else ""
            self._log(
                f"Q combo={combinacion_id} {qi}/{n_q} status={q_status} "
                f"kw_hits={hits_kw}/{len(keywords)} score={sc:.4f} time_s={elapsed:.2f} "
                f"| {preview}{tail}"
            )

        self._cleanup_chroma_dir(Path(persist_dir))

        if combo_failed:
            t_avg = sum(times) / len(times) if times else -1.0
            return {
                "timestamp": row_ts,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "temperature": temperature,
                "score_promedio": -1.0,
                "score_minimo": -1.0,
                "score_maximo": -1.0,
                "tiempo_promedio_segundos": round(t_avg, 4) if t_avg >= 0 else -1.0,
                "mejor_pregunta": "",
                "peor_pregunta": "",
                "combinacion_id": combinacion_id,
            }

        scores_only = [p[0] for p in per_q_scores]
        avg = sum(scores_only) / len(scores_only) if scores_only else 0.0
        mn = min(scores_only) if scores_only else 0.0
        mx = max(scores_only) if scores_only else 0.0
        t_avg = sum(times) / len(times) if times else 0.0

        best_q = max(per_q_scores, key=lambda x: x[0])[1] if per_q_scores else ""
        worst_q = min(per_q_scores, key=lambda x: x[0])[1] if per_q_scores else ""

        return {
            "timestamp": row_ts,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "temperature": temperature,
            "score_promedio": round(avg, 4),
            "score_minimo": round(mn, 4),
            "score_maximo": round(mx, 4),
            "tiempo_promedio_segundos": round(t_avg, 4),
            "mejor_pregunta": best_q[:200],
            "peor_pregunta": worst_q[:200],
            "combinacion_id": combinacion_id,
        }

    def _append_csv_row(self, path: Path, row: dict[str, Any], write_header: bool) -> None:
        fieldnames = [
            "timestamp",
            "chunk_size",
            "chunk_overlap",
            "top_k",
            "temperature",
            "score_promedio",
            "score_minimo",
            "score_maximo",
            "tiempo_promedio_segundos",
            "mejor_pregunta",
            "peor_pregunta",
            "combinacion_id",
        ]
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})

    def _write_best_config(
        self,
        csv_path: Path,
        total_full_grid: int,
        n_tried: int,
    ) -> None:
        rows: list[dict[str, Any]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    sp = float(row["score_promedio"])
                except ValueError:
                    continue
                if sp < 0:
                    continue
                rows.append(row)
        if not rows:
            summary = "=== MEJOR CONFIGURACIÓN ENCONTRADA ===\nNo hay filas válidas (score >= 0) en el CSV.\n"
            out_json = self.output_dir / f"best_config_{self._run_timestamp}.json"
            out_json.write_text(
                json.dumps(
                    {"executive_summary": summary, "best": None},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(summary)
            return

        best = max(rows, key=lambda x: float(x["score_promedio"]))
        summary_lines = [
            "=== MEJOR CONFIGURACIÓN ENCONTRADA ===",
            f"chunk_size: {best['chunk_size']}",
            f"chunk_overlap: {best['chunk_overlap']}",
            f"top_k: {best['top_k']}",
            f"temperature: {best['temperature']}",
            f"Score promedio: {best['score_promedio']}",
            f"Tiempo promedio de respuesta: {best['tiempo_promedio_segundos']}s",
            f"Evaluado sobre: {len(GOLDEN_DATASET)} preguntas del golden dataset",
            f"Total combinaciones probadas: {n_tried}/{total_full_grid}",
        ]
        summary = "\n".join(summary_lines) + "\n"
        payload = {
            "best": {k: best[k] for k in best},
            "executive_summary": summary,
            "csv_path": str(csv_path),
            "golden_questions": len(GOLDEN_DATASET),
            "combinations_tried": n_tried,
            "full_grid_size": total_full_grid,
        }
        out_json = self.output_dir / f"best_config_{self._run_timestamp}.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(summary)

    def run(
        self,
        max_combinations: int | None = None,
    ) -> Path:
        """Execute evaluation; returns path to results CSV."""
        full_grid = expand_param_grid()
        total_full = len(full_grid)

        if self.mode == "full":
            combinations = list(full_grid)
        else:
            combinations = sample_combinations(full_grid, 20, self.seed)

        if max_combinations is not None:
            combinations = combinations[: max(0, max_combinations)]

        if self.mode == "quick":
            print(
                f"Semilla muestreo quick: {self.seed} "
                f"(repetir el mismo orden: --seed {self.seed} o RAG_EVAL_SEED={self.seed})",
                flush=True,
            )

        completed: list[int] = []
        write_header = True
        log_path: Path

        if self.fresh and self._checkpoint_path.is_file():
            self._checkpoint_path.unlink()

        ck = None if self.fresh else self._load_checkpoint()
        ck_combos = ck.get("combinations") if ck else None
        if (
            ck
            and ck.get("mode") == self.mode
            and isinstance(ck_combos, list)
            and _combinations_match(ck_combos, combinations)  # type: ignore[arg-type]
            and Path(ck.get("csv_path", "")).is_file()
        ):
            self._run_timestamp = ck["run_timestamp"]
            self._csv_path = Path(ck["csv_path"])
            log_path = Path(ck["log_path"])
            completed = sorted({int(x) for x in ck.get("completed_combination_ids", [])})
            write_header = False
            self._log_fp = self._open_log(log_path)
            self._log(f"RESUME run {self._run_timestamp}, completed={completed}")
            if self.mode == "quick":
                self._log(
                    f"Muestreo quick (checkpoint) seed={ck.get('seed', self.seed)}"
                )
        else:
            if ck and not self.fresh:
                print(
                    "Aviso: checkpoint incompatible o CSV ausente; iniciando corrida nueva.",
                    file=sys.stderr,
                )
            self._run_timestamp = self._new_run_timestamp()
            self._csv_path = self.output_dir / f"results_{self._run_timestamp}.csv"
            log_path = self.output_dir / f"eval_log_{self._run_timestamp}.txt"
            self._log_fp = self._open_log(log_path)
            write_header = True
            completed = []
            if self.mode == "quick":
                self._log(
                    f"Muestreo quick: seed={self.seed} "
                    f"(repetir: --seed {self.seed} o RAG_EVAL_SEED={self.seed})"
                )

        assert self._csv_path is not None
        csv_path = self._csv_path

        if write_header and not csv_path.is_file():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "chunk_size",
                        "chunk_overlap",
                        "top_k",
                        "temperature",
                        "score_promedio",
                        "score_minimo",
                        "score_maximo",
                        "tiempo_promedio_segundos",
                        "mejor_pregunta",
                        "peor_pregunta",
                        "combinacion_id",
                    ],
                )
                w.writeheader()

        resolved = resolve_legal_md_paths(self.pdf_dir)
        self._log(
            "Corpus legal: "
            + ", ".join(f"{p.name} ({label})" for p, label, _ in resolved)
        )
        docs = load_legal_corpus(self.pdf_dir)
        self._log(f"Directorio corpus: {self.pdf_dir} — páginas cargadas={len(docs)}")

        n_total = len(combinations)
        combo_id_list = list(range(1, n_total + 1))
        pending_idx = [i for i, cid in enumerate(combo_id_list) if cid not in completed]

        iterator: Iterator[int]
        if _HAS_TQDM:
            iterator = tqdm(pending_idx, desc="RAG eval", unit="combo")
        else:
            iterator = iter(pending_idx)

        t_run_start = time.perf_counter()
        done_count = len(completed)

        for idx in iterator:
            combinacion_id = combo_id_list[idx]
            combo = combinations[idx]
            t0 = time.perf_counter()
            row = self._evaluate_one_combo(combo, combinacion_id, docs)
            elapsed_combo = time.perf_counter() - t0

            self._append_csv_row(csv_path, row, write_header=False)
            write_header = False
            completed.append(combinacion_id)
            done_count += 1
            self._save_checkpoint(combinations, completed, csv_path, log_path)

            remaining = n_total - done_count
            if remaining > 0 and done_count > 0:
                avg_per = (time.perf_counter() - t_run_start) / done_count
                eta_sec = int(avg_per * remaining)
                eta_min = eta_sec // 60
            else:
                eta_min = 0

            msg = (
                f"Combinación {done_count}/{n_total} — "
                f"Score: {row['score_promedio']} — ETA: ~{eta_min} min"
            )
            print(msg)
            self._log(
                f"{msg} | combo_id={combinacion_id} | wall_combo={elapsed_combo:.1f}s"
            )

        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None

        if self._checkpoint_path.is_file():
            self._checkpoint_path.unlink(missing_ok=True)

        self._write_best_config(csv_path, total_full_grid=total_full, n_tried=n_total)
        return csv_path


def latest_results_csv(output_dir: Path | None = None) -> Path | None:
    if output_dir is not None:
        base = Path(output_dir)
    else:
        base = _project_root() / "outputs" / "rag_eval"
    if not base.is_dir():
        return None
    files = sorted(base.glob("results_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def print_results_only() -> int:
    p = latest_results_csv()
    if not p:
        print("No hay results_*.csv en outputs/rag_eval/")
        return 1
    print(f"Último CSV: {p}")
    best_jsons = sorted(
        p.parent.glob("best_config_*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    if best_jsons:
        print(f"Último best_config: {best_jsons[0]}")
        data = json.loads(best_jsons[0].read_text(encoding="utf-8"))
        print(data.get("executive_summary", ""))
    return 0
