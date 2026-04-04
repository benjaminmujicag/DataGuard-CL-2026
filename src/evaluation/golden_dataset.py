"""Golden dataset for RAG evaluation — audit-oriented questions with precise keywords.

# Revision: 2026-04-03
# Improved: keywords now include article-specific terms and legal phrases that
# discriminate real answers from generic LLM output. Each entry has a ``weight``
# (1.0 = standard, 2.0 = critical for audit) so the evaluator can compute
# weighted scores.

Each entry has:
  - question: what rag_classify or an auditor would actually ask
  - expected_keywords: terms that MUST appear in a correct answer (specific to articles)
  - required_citations: article references the answer MUST cite (e.g. "artículo 16 bis")
  - weight: importance multiplier for scoring (1.0 = normal, 2.0 = critical)
"""

from __future__ import annotations

GOLDEN_DATASET: list[dict[str, str | list[str] | float]] = [
    # ---- Per-category queries (replicating rag_classify prompt format) ----
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos identificadores?"
        ),
        "expected_keywords": [
            "dato personal",
            "responsable de datos",
            "base de licitud",
            "consentimiento del titular",
            "deber de secreto",
        ],
        "required_citations": ["artículo 3"],
        "weight": 1.0,
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos de salud (Sensibles)?"
        ),
        "expected_keywords": [
            "datos sensibles",
            "dato personal sensible",
            "consentimiento expreso",
            "prohibición general",
            "salud del titular",
        ],
        "required_citations": ["artículo 16 bis"],
        "weight": 2.0,
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos financieros?"
        ),
        "expected_keywords": [
            "dato personal",
            "principio de finalidad",
            "medidas de seguridad",
            "proporcionalidad",
            "deber de confidencialidad",
        ],
        "required_citations": ["artículo 3"],
        "weight": 1.5,
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos de contacto?"
        ),
        "expected_keywords": [
            "base de licitud",
            "principio de finalidad",
            "consentimiento",
            "comunicación de datos",
            "derecho de oposición",
        ],
        "required_citations": ["artículo 12"],
        "weight": 1.0,
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos biométricos?"
        ),
        "expected_keywords": [
            "dato biométrico",
            "dato personal sensible",
            "prohibición general de tratamiento",
            "consentimiento expreso",
            "excepciones legales",
        ],
        "required_citations": ["artículo 16 bis"],
        "weight": 2.0,
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Potencial dato de menores?"
        ),
        "expected_keywords": [
            "niño",
            "interés superior del niño",
            "representante legal",
            "consentimiento",
            "menor de 14",
        ],
        "required_citations": ["artículo 16 quinquies"],
        "weight": 2.0,
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Datos de geolocalización?"
        ),
        "expected_keywords": [
            "dato personal",
            "principio de proporcionalidad",
            "finalidad específica",
            "minimización de datos",
            "consentimiento del titular",
        ],
        "required_citations": ["artículo 3"],
        "weight": 1.0,
    },
    {
        "question": (
            "Según la legislación chilena de protección de datos personales (Ley 21.719 "
            "y relacionadas), ¿qué obligaciones o restricciones aplican al tratamiento "
            "de datos clasificados como: Logs y auditoría?"
        ),
        "expected_keywords": [
            "registro de acceso",
            "deber de seguridad",
            "proporcionalidad",
            "responsable de datos",
            "medidas técnicas",
        ],
        "required_citations": ["artículo 14 quinquies"],
        "weight": 1.0,
    },
    # ---- Audit-specific questions a CISO would ask ----
    {
        "question": (
            "¿Qué medidas de seguridad exige la ley para bases de datos que contienen "
            "datos personales?"
        ),
        "expected_keywords": [
            "medidas de seguridad",
            "deber de seguridad",
            "confidencialidad",
            "integridad",
            "disponibilidad",
        ],
        "required_citations": ["artículo 14 quinquies"],
        "weight": 2.0,
    },
    {
        "question": (
            "¿Qué es el responsable de datos y qué obligaciones tiene respecto de su "
            "base de datos?"
        ),
        "expected_keywords": [
            "responsable de datos",
            "determina los fines",
            "deber de seguridad",
            "registro de actividades",
            "delegado de protección",
        ],
        "required_citations": ["artículo 14"],
        "weight": 1.5,
    },
    {
        "question": (
            "¿Cuándo se requiere evaluación de impacto en el tratamiento de datos "
            "personales, según la Ley 21.719?"
        ),
        "expected_keywords": [
            "evaluación de impacto",
            "alto riesgo",
            "datos sensibles",
            "tratamiento a gran escala",
            "medidas de mitigación",
        ],
        "required_citations": ["artículo 14 septies"],
        "weight": 2.0,
    },
    {
        "question": (
            "¿Qué principios debe respetar el tratamiento de datos en una base de datos "
            "según la legislación chilena?"
        ),
        "expected_keywords": [
            "principio de finalidad",
            "proporcionalidad",
            "licitud del tratamiento",
            "calidad de los datos",
            "principio de seguridad",
        ],
        "required_citations": ["artículo 3"],
        "weight": 1.5,
    },
    {
        "question": (
            "¿Qué obligaciones aplican cuando una columna de una base de datos almacena "
            "números de cédula de identidad o RUT?"
        ),
        "expected_keywords": [
            "dato personal",
            "identificador único",
            "responsable de datos",
            "base de licitud",
            "deber de seguridad",
        ],
        "required_citations": ["artículo 3", "artículo 14 quinquies"],
        "weight": 1.5,
    },
    {
        "question": (
            "¿Qué restricciones aplican al tratamiento de datos sensibles almacenados "
            "en una base de datos según la Ley 21.719?"
        ),
        "expected_keywords": [
            "datos sensibles",
            "prohibición general",
            "consentimiento expreso",
            "excepciones taxativas",
            "categorías especiales",
        ],
        "required_citations": ["artículo 16 bis"],
        "weight": 2.0,
    },
    {
        "question": (
            "¿Qué dice la ley sobre el almacenamiento de datos de salud en sistemas "
            "informáticos y bases de datos?"
        ),
        "expected_keywords": [
            "datos relativos a la salud",
            "dato sensible",
            "consentimiento expreso",
            "profesional de la salud",
            "medidas de seguridad",
        ],
        "required_citations": ["artículo 16 bis"],
        "weight": 2.0,
    },
    {
        "question": (
            "¿Qué obligaciones tiene el mandatario (encargado de tratamiento) que procesa "
            "datos personales por cuenta de otro?"
        ),
        "expected_keywords": [
            "mandatario",
            "encargado de tratamiento",
            "instrucciones del responsable",
            "contrato de mandato",
            "deber de confidencialidad",
        ],
        "required_citations": ["artículo 15 bis"],
        "weight": 1.5,
    },
    {
        "question": (
            "¿Cuándo puede una organización tratar datos personales sin obtener "
            "consentimiento del titular?"
        ),
        "expected_keywords": [
            "sin consentimiento",
            "obligación legal",
            "ejecución de contrato",
            "interés legítimo",
            "fuentes de acceso público",
        ],
        "required_citations": ["artículo 13"],
        "weight": 1.5,
    },
    {
        "question": (
            "¿Qué derechos del titular deben poder ejercerse sobre los datos personales "
            "almacenados en una base de datos?"
        ),
        "expected_keywords": [
            "derecho de acceso",
            "rectificación",
            "supresión",
            "portabilidad",
            "derecho de oposición",
        ],
        "required_citations": ["artículo 5", "artículo 6"],
        "weight": 1.5,
    },
    {
        "question": (
            "¿Qué obligaciones aplican cuando una base de datos contiene datos de personas "
            "menores de 14 años?"
        ),
        "expected_keywords": [
            "niño, niña o adolescente",
            "representante legal",
            "consentimiento del representante",
            "interés superior del niño",
            "menor de 14 años",
        ],
        "required_citations": ["artículo 16 quinquies"],
        "weight": 2.0,
    },
    {
        "question": (
            "¿Qué infracciones en el tratamiento de datos personales en una base de datos "
            "se consideran graves y deben reportarse en una auditoría?"
        ),
        "expected_keywords": [
            "infracción grave",
            "infracción gravísima",
            "multa",
            "Agencia de Protección de Datos",
            "vulneración de seguridad",
        ],
        "required_citations": ["artículo 34 bis", "artículo 34 ter"],
        "weight": 2.0,
    },
]
