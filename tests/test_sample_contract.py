"""
Contract tests verifying exact conformity with sample_output_format.csv.
"""

import os
import unittest
import pandas as pd

from src.cli import run_pipeline


class TestSampleContract(unittest.TestCase):
    """Verifies that assistant output strictly satisfies the sample_output_format contract."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        cls.shipments = os.path.join(cls.data_dir, "shipment_records.csv")
        cls.notes = os.path.join(cls.data_dir, "context_notes.csv")
        cls.sample_path = os.path.join(cls.data_dir, "sample_output_format.csv")
        cls.test_output = "temp_test_output.csv"

        # Generate fresh output
        run_pipeline(
            shipments_path=cls.shipments,
            notes_path=cls.notes,
            output_path=cls.test_output,
            quiet=True,
        )

        cls.output_df = pd.read_csv(cls.test_output)

        # Parse sample file with comma handling
        sample_rows = []
        with open(cls.sample_path, "r", encoding="utf-8") as f:
            headers = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    sample_rows.append({
                        "route": parts[0],
                        "week_of": parts[1],
                        "cost_per_tonne_km": float(parts[2]),
                        "vs_own_history": parts[3],
                        "vs_similar_routes": parts[4],
                        "flagged": parts[5],
                        "matched_note_id": parts[6] if parts[6] else "",
                        "reason": ",".join(parts[7:]),
                    })
        cls.sample_df = pd.DataFrame(sample_rows)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output):
            os.remove(cls.test_output)

    def test_column_names_and_order(self):
        """Column names, order, and types in sample_output_format.csv are the contract."""
        expected_columns = [
            "route", "week_of", "cost_per_tonne_km", "vs_own_history",
            "vs_similar_routes", "flagged", "matched_note_id", "reason"
        ]
        self.assertEqual(list(self.output_df.columns), expected_columns)

    def test_sample_row_delhi_jaipur(self):
        """Sample Row 1: Delhi-Jaipur on 2024-11-11."""
        row = self.output_df[
            (self.output_df["route"] == "Delhi-Jaipur") & (self.output_df["week_of"] == "2024-11-11")
        ]
        self.assertEqual(len(row), 1, "Delhi-Jaipur 2024-11-11 must be present in output")
        r = row.iloc[0]
        self.assertAlmostEqual(float(r["cost_per_tonne_km"]), 4.17, places=2)
        self.assertEqual(r["vs_own_history"], "+35.5% vs this route's past average")
        self.assertEqual(r["vs_similar_routes"], "+21.0% vs similar-length routes this week")
        self.assertEqual(r["flagged"], "Yes")
        self.assertTrue(pd.isna(r["matched_note_id"]) or r["matched_note_id"] == "")
        self.assertIn("No matching note found", r["reason"])

    def test_sample_row_ahmedabad_mumbai(self):
        """Sample Row 2: Ahmedabad-Mumbai on 2025-01-20."""
        row = self.output_df[
            (self.output_df["route"] == "Ahmedabad-Mumbai") & (self.output_df["week_of"] == "2025-01-20")
        ]
        self.assertEqual(len(row), 1, "Ahmedabad-Mumbai 2025-01-20 must be present in output")
        r = row.iloc[0]
        self.assertAlmostEqual(float(r["cost_per_tonne_km"]), 3.29, places=2)
        self.assertEqual(r["vs_own_history"], "+29.5% vs this route's past average")
        self.assertEqual(r["vs_similar_routes"], "+22.5% vs similar-length routes this week")
        self.assertEqual(r["flagged"], "No (justified)")
        self.assertEqual(r["matched_note_id"], "N002")
        self.assertIn("Matches note N002", r["reason"])

    def test_sample_row_mumbai_pune(self):
        """Sample Row 3: Mumbai-Pune on 2025-09-15."""
        row = self.output_df[
            (self.output_df["route"] == "Mumbai-Pune") & (self.output_df["week_of"] == "2025-09-15")
        ]
        self.assertEqual(len(row), 1, "Mumbai-Pune 2025-09-15 must be present in output")
        r = row.iloc[0]
        self.assertAlmostEqual(float(r["cost_per_tonne_km"]), 3.98, places=2)
        self.assertEqual(r["vs_own_history"], "+9.2% vs this route's past average")
        self.assertEqual(r["vs_similar_routes"], "+23.6% vs similar-length routes this week")
        self.assertEqual(r["flagged"], "Yes")
        self.assertTrue(pd.isna(r["matched_note_id"]) or r["matched_note_id"] == "")
        self.assertIn("The closest note (N006", r["reason"])


if __name__ == "__main__":
    unittest.main()
