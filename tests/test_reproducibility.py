"""
Reproducibility tests verifying identical outputs across 3 untouched runs.
"""

import os
import unittest
import pandas as pd

from src.cli import run_pipeline


class TestReproducibility(unittest.TestCase):
    """Verifies that 3 untouched runs on the same input produce identical flags, numbers, and cited notes."""

    def test_three_untouched_runs_match(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        shipments = os.path.join(data_dir, "shipment_records.csv")
        notes = os.path.join(data_dir, "context_notes.csv")

        run_files = ["repro_run_1.csv", "repro_run_2.csv", "repro_run_3.csv"]
        contents = []

        try:
            for rf in run_files:
                run_pipeline(
                    shipments_path=shipments,
                    notes_path=notes,
                    output_path=rf,
                    quiet=True,
                )
                with open(rf, "r", encoding="utf-8") as f:
                    contents.append(f.read())

            # Assert identical output text
            self.assertEqual(contents[0], contents[1], "Run 1 and Run 2 outputs differ!")
            self.assertEqual(contents[1], contents[2], "Run 2 and Run 3 outputs differ!")

            # Verify DataFrame parsed data equality
            df1 = pd.read_csv(run_files[0])
            df2 = pd.read_csv(run_files[1])
            df3 = pd.read_csv(run_files[2])

            pd.testing.assert_frame_equal(df1, df2)
            pd.testing.assert_frame_equal(df2, df3)

        finally:
            for rf in run_files:
                if os.path.exists(rf):
                    os.remove(rf)


if __name__ == "__main__":
    unittest.main()
