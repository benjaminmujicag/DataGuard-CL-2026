"""Golden dataset for RAG evaluation — audit-oriented questions.

# Review: opus-4.6 · 2026-04-03 (full replacement)
# Previous dataset tested general legal knowledge (ARCO rights, deadlines, sanctions).
# These 20 questions replicate the actual query format used by rag_classify in production,
# covering the 8 column categories from classify_column + cross-cutting audit scenarios.
# Keywords are grounded in ley_21719.md and ley_mixta.md (leyes_base corpus).

Each entry has:
  - question: what rag_classify or an auditor would actually ask
  - expected_keywords: terms that MUST appear in a correct answer
"""

from __future__ import annotations

GOLDEN_DATASET: list[dict[str, str | list[str]]] = [
    # ---- Per-category queries (replicating rag_classify prompt format) ----
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos identificadores?"
        ),
        "expected_keywords": [
            "responsable",
            "consentimiento",
            "identificación",
            "tratamiento",
            "titular",
        ],
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos de salud (Sensibles)?"
        ),
        "expected_keywords": [
            "sensibles",
            "salud",
            "consentimiento",
            "prohibición",
            "expreso",
        ],
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos financieros?"
        ),
        "expected_keywords": [
            "financiero",
            "seguridad",
            "proporcionalidad",
            "finalidad",
            "responsable",
        ],
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos de contacto?"
        ),
        "expected_keywords": [
            "consentimiento",
            "finalidad",
            "comunicación",
            "licitud",
            "titular",
        ],
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos biométricos?"
        ),
        "expected_keywords": [
            "biométrico",
            "sensible",
            "prohibición",
            "consentimiento",
            "excepciones",
        ],
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Potencial dato de menores?"
        ),
        "expected_keywords": [
            "menores",
            "interés superior",
            "representante",
            "consentimiento",
            "edad",
        ],
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos de geolocalización?"
        ),
        "expected_keywords": [
            "localización",
            "proporcionalidad",
            "finalidad",
            "minimización",
            "consentimiento",
        ],
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Logs y auditoría?"
        ),
        "expected_keywords": [
            "acceso",
            "registro",
            "seguridad",
            "proporcionalidad",
            "responsable",
        ],
    },
    # ---- Audit-specific questions a CISO would ask ----
    {
        "question": (
            "¿Qué medidas de seguridad exige la ley para bases de datos que contienen "
            "datos personales?"
        ),
        "expected_keywords": [
            "seguridad",
            "medidas",
            "acceso",
            "integridad",
            "confidencialidad",
        ],
    },
    {
        "question": (
            "¿Qué es el responsable de datos y qué obligaciones tiene respecto de su "
            "base de datos?"
        ),
        "expected_keywords": [
            "responsable",
            "obligaciones",
            "tratamiento",
            "seguridad",
            "titular",
        ],
    },
    {
        "question": (
            "¿Cuándo se requiere evaluación de impacto en el tratamiento de datos "
            "personales, según la Ley 21.719?"
        ),
        "expected_keywords": [
            "evaluación de impacto",
            "riesgo",
            "datos sensibles",
            "responsable",
            "tratamiento",
        ],
    },
    {
        "question": (
            "¿Qué principios debe respetar el tratamiento de datos en una base de datos "
            "según la legislación chilena?"
        ),
        "expected_keywords": [
            "finalidad",
            "proporcionalidad",
            "licitud",
            "minimización",
            "exactitud",
        ],
    },
    {
        "question": (
            "¿Qué obligaciones aplican cuando una columna de una base de datos almacena "
            "números de cédula de identidad o RUT?"
        ),
        "expected_keywords": [
            "identificador",
            "responsable",
            "tratamiento",
            "consentimiento",
            "seguridad",
        ],
    },
    {
        "question": (
            "¿Qué restricciones aplican al tratamiento de datos sensibles almacenados "
            "en una base de datos según la Ley 21.719?"
        ),
        "expected_keywords": [
            "sensibles",
            "prohibición",
            "expreso",
            "consentimiento",
            "excepciones",
        ],
    },
    {
        "question": (
            "¿Qué dice la ley sobre el almacenamiento de datos de salud en sistemas "
            "informáticos y bases de datos?"
        ),
        "expected_keywords": [
            "salud",
            "almacenamiento",
            "seguridad",
            "sensibles",
            "responsable",
        ],
    },
    {
        "question": (
            "¿Qué obligaciones tiene el mandatario (encargado de tratamiento) que procesa "
            "datos personales por cuenta de otro?"
        ),
        "expected_keywords": [
            "mandatario",
            "responsable",
            "instrucciones",
            "seguridad",
            "contrato",
        ],
    },
    {
        "question": (
            "¿Cuándo puede una organización tratar datos personales sin obtener "
            "consentimiento del titular?"
        ),
        "expected_keywords": [
            "consentimiento",
            "obligación legal",
            "contrato",
            "interés legítimo",
            "excepción",
        ],
    },
    {
        "question": (
            "¿Qué derechos del titular deben poder ejercerse sobre los datos personales "
            "almacenados en una base de datos?"
        ),
        "expected_keywords": [
            "acceso",
            "rectificación",
            "cancelación",
            "portabilidad",
            "titular",
        ],
    },
    {
        "question": (
            "¿Qué obligaciones aplican cuando una base de datos contiene datos de personas "
            "menores de 14 años?"
        ),
        "expected_keywords": [
            "menores",
            "representante legal",
            "consentimiento",
            "interés superior",
            "edad",
        ],
    },
    {
        "question": (
            "¿Qué infracciones en el tratamiento de datos personales en una base de datos "
            "se consideran graves y deben reportarse en una auditoría?"
        ),
        "expected_keywords": [
            "infracción",
            "grave",
            "sanción",
            "agencia",
            "tratamiento",
        ],
    },
]
