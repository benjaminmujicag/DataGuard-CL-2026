"""Tests for LangGraph audit workflow (RAG mocked)."""

import os
import unittest
from unittest.mock import patch

from src.graph.workflow import (
    GRAPH_NODE_ORDER,
    create_audit_workflow,
    run_graph_audit,
    run_graph_audit_traced,
)


class TestWorkflow(unittest.TestCase):
    """Smoke tests for deterministic graph path."""

    @patch("src.graph.workflow.query_legal", return_value="Articulo ficticio para prueba.")
    def test_invoke_returns_report_structure(self, _mock_q: object) -> None:
        schema = os.path.join("data", "sample_schema.sql")
        if not os.path.exists(schema):
            self.skipTest("sample_schema.sql missing")

        app = create_audit_workflow()
        final = app.invoke({"schema_path": schema, "errors": [], "columns": []})
        self.assertIn("report", final)
        rep = final["report"]
        self.assertIn("resumen", rep)
        self.assertIn("hallazgos", rep)
        self.assertGreater(rep["resumen"].get("total_columnas", 0), 0)

    @patch("src.graph.workflow.query_legal", return_value="stub")
    def test_run_graph_audit_helper(self, _mock_q: object) -> None:
        schema = os.path.join("data", "sample_schema.sql")
        if not os.path.exists(schema):
            self.skipTest("sample_schema.sql missing")
        rep = run_graph_audit(schema)
        self.assertNotIn("error", rep)
        self.assertTrue(rep.get("hallazgos"))

    @patch("src.graph.workflow.query_legal", return_value="stub legal")
    def test_run_graph_audit_traced_logs_nodes(self, _mock_q: object) -> None:
        schema = os.path.join("data", "sample_schema.sql")
        if not os.path.exists(schema):
            self.skipTest("sample_schema.sql missing")

        rep, steps = run_graph_audit_traced(schema)
        node_ids = [s[0] for s in steps]
        self.assertEqual(node_ids, list(GRAPH_NODE_ORDER))
        self.assertIn("resumen", rep)
        self.assertGreater(len(rep.get("hallazgos", [])), 0)


if __name__ == "__main__":
    unittest.main()
