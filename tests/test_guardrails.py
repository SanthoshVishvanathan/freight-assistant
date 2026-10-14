"""
Anti-hallucination guardrail validation tests.

Tests:
1. Negative-clause notes (N004, N005, N010, N006) are NEVER cited as justifications.
2. Temporal window boundaries strictly enforced (N001 only during flood window).
3. Corridor mismatch rejection.
4. Spot-check audit against all generated output rows.
"""

import os
import unittest
import pandas as pd

from src.cli import run_pipeline
from src.core.detector import RouteAnomalyCandidate
from src.rag.vector_store import ContextNotesVectorStore
from src.rag.retriever import ContextRetriever
from src.rag.guardrails import GuardrailValidator


class TestGuardrails(unittest.TestCase):
    """Guarantees anti-hallucination rules and business logic integrity."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        cls.notes_path = os.path.join(cls.data_dir, "context_notes.csv")
        cls.vector_store = ContextNotesVectorStore(cls.notes_path)
        cls.retriever = ContextRetriever(cls.vector_store)

        cls.shipments = os.path.join(cls.data_dir, "shipment_records.csv")
        cls.output_file = "temp_guardrails_test_output.csv"
        cls.exps = run_pipeline(
            shipments_path=cls.shipments,
            notes_path=cls.notes_path,
            output_path=cls.output_file,
            quiet=True,
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.output_file):
            os.remove(cls.output_file)

    def test_n004_toll_plaza_never_justifies(self):
        """Note N004 states affected routes are NOT part of this dataset -> must never justify."""
        for exp in self.exps:
            self.assertNotEqual(exp.matched_note_id, "N004", "N004 must NEVER be cited as justification!")

    def test_n005_maintenance_never_justifies(self):
        """Note N005 states costs were NOT significantly affected -> must never justify."""
        for exp in self.exps:
            self.assertNotEqual(exp.matched_note_id, "N005", "N005 must NEVER be cited as justification!")

    def test_n006_stable_demand_never_justifies(self):
        """Note N006 states stable demand with no disruptions -> must never justify."""
        for exp in self.exps:
            self.assertNotEqual(exp.matched_note_id, "N006", "N006 must NEVER be cited as justification!")

    def test_n010_tracking_mandate_never_justifies(self):
        """Note N010 states costs absorbed without rate change -> must never justify."""
        for exp in self.exps:
            self.assertNotEqual(exp.matched_note_id, "N010", "N010 must NEVER be cited as justification!")

    def test_n001_flood_temporal_window_strictness(self):
        """N001 only justifies Chennai-Bangalore during Feb 24 to Mar 8 flood window."""
        for exp in self.exps:
            if exp.matched_note_id == "N001":
                self.assertEqual(exp.route, "Chennai-Bangalore")
                self.assertIn(exp.week_of, ["2025-02-24", "2025-03-03"],
                              f"N001 cited outside flood window on {exp.week_of}!")

    def test_n002_festival_surcharge_strictness(self):
        """N002 only justifies Ahmedabad-Mumbai during the festival week."""
        for exp in self.exps:
            if exp.matched_note_id == "N002":
                self.assertEqual(exp.route, "Ahmedabad-Mumbai")
                self.assertEqual(exp.week_of, "2025-01-20")

    def test_all_justified_verdicts_cite_valid_note_id(self):
        """Every row marked 'No (justified)' MUST cite a non-empty matched_note_id."""
        for exp in self.exps:
            if exp.flagged == "No (justified)":
                self.assertTrue(bool(exp.matched_note_id), f"Row {exp.route} {exp.week_of} is justified but has blank note ID!")
            elif exp.flagged == "Yes":
                self.assertEqual(exp.matched_note_id, "", f"Row {exp.route} {exp.week_of} is unexplained but has note ID {exp.matched_note_id}!")


if __name__ == "__main__":
    unittest.main()
