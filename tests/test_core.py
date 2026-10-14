"""
Unit tests for core mathematical aggregations, rolling windows, and peer baselines.
"""

import datetime
import unittest
import pandas as pd

from src.core.aggregator import ShipmentAggregator
from src.core.baselines import BaselineCalculator


class TestCoreAggregation(unittest.TestCase):
    """Verifies weekly Monday-Sunday grouping and cost per tonne-km formula."""

    def test_week_of_monday_calculation(self):
        # 2024-01-01 was a Monday -> week_of should be 2024-01-01
        self.assertEqual(ShipmentAggregator.calculate_week_of(datetime.date(2024, 1, 1)), "2024-01-01")
        # 2024-01-05 was a Friday -> week_of should be 2024-01-01
        self.assertEqual(ShipmentAggregator.calculate_week_of(datetime.date(2024, 1, 5)), "2024-01-01")
        # 2024-01-07 was a Sunday -> week_of should be 2024-01-01
        self.assertEqual(ShipmentAggregator.calculate_week_of(datetime.date(2024, 1, 7)), "2024-01-01")
        # 2024-01-08 was next Monday -> week_of should be 2024-01-08
        self.assertEqual(ShipmentAggregator.calculate_week_of(datetime.date(2024, 1, 8)), "2024-01-08")

    def test_cost_per_tonne_km_formula(self):
        """
        Brief definition:
        Cost per tonne per km = total freight cost / (quantity carried in tonnes x distance travelled in km).
        """
        sample_df = pd.DataFrame([
            {
                "shipment_id": "S1", "origin": "A", "destination": "B", "route_type": "Short",
                "material": "Steel", "quantity_tonnes": 10.0, "distance_km": 100.0,
                "freight_cost_inr": 3000.0, "shipment_date": "2024-01-02"
            },
            {
                "shipment_id": "S2", "origin": "A", "destination": "B", "route_type": "Short",
                "material": "Textiles", "quantity_tonnes": 20.0, "distance_km": 100.0,
                "freight_cost_inr": 6000.0, "shipment_date": "2024-01-04"
            }
        ])

        agg = ShipmentAggregator.aggregate(sample_df)
        self.assertEqual(len(agg), 1)
        row = agg.iloc[0]
        # Total cost = 9000
        # Total tonne-km = 10*100 + 20*100 = 1000 + 2000 = 3000
        # Cost per tonne-km = 9000 / 3000 = 3.00
        self.assertAlmostEqual(row["cost_per_tonne_km"], 3.0, places=4)


class TestBaselinesCalculation(unittest.TestCase):
    """Verifies 8-week trailing own rolling average and peer route baseline."""

    def test_zero_lookahead_and_window_size(self):
        """Ensures rolling average strictly excludes current week and uses max 8 prior weeks."""
        rows = []
        # Create 10 consecutive weeks for Route X
        for w_idx in range(10):
            mon = (datetime.date(2024, 1, 1) + datetime.timedelta(weeks=w_idx)).strftime("%Y-%m-%d")
            rows.append({
                "route": "CityA-CityB", "origin": "CityA", "destination": "CityB",
                "route_type": "Short", "week_of": mon,
                "cost_per_tonne_km": float(w_idx + 1),  # costs 1.0, 2.0, ..., 10.0
                "total_freight_cost": 1000.0, "total_tonne_km": 1000.0, "shipment_count": 1
            })

        df = pd.DataFrame(rows)
        baselines = BaselineCalculator.compute_all(df, rolling_window_weeks=8)

        # For week 0 (first week), prior count should be 0
        w0 = baselines[baselines["week_of"] == "2024-01-01"].iloc[0]
        self.assertEqual(w0["prior_weeks_available"], 0)
        self.assertTrue(pd.isna(w0["pct_vs_own_history"]))

        # For week 9 (10th week, cost=10.0):
        # Strictly prior 8 weeks are weeks 1 to 8 (costs 2, 3, 4, 5, 6, 7, 8, 9)
        # Average = (2+3+4+5+6+7+8+9)/8 = 44/8 = 5.5
        # Current cost = 10.0 -> pct diff = (10 - 5.5)/5.5 * 100 = 4.5/5.5 * 100 = +81.8%
        w9 = baselines.iloc[9]
        self.assertEqual(w9["prior_weeks_available"], 8)
        self.assertAlmostEqual(w9["pct_vs_own_history"], (4.5 / 5.5) * 100.0, places=1)
        self.assertIn("+81.8% vs this route's past average", w9["vs_own_history"])

    def test_peer_routes_exclusion_of_self(self):
        """Ensures peer average excludes the target route itself."""
        rows = [
            {"route": "Route1", "route_type": "Medium", "week_of": "2024-01-01", "cost_per_tonne_km": 2.0},
            {"route": "Route2", "route_type": "Medium", "week_of": "2024-01-01", "cost_per_tonne_km": 3.0},
            {"route": "Route3", "route_type": "Medium", "week_of": "2024-01-01", "cost_per_tonne_km": 5.0},
        ]
        df = pd.DataFrame(rows)
        baselines = BaselineCalculator.compute_all(df)

        # For Route1 (cost=2.0), peers are Route2 (3.0) and Route3 (5.0) -> mean = 4.0
        # pct = (2.0 - 4.0) / 4.0 * 100 = -50.0%
        r1 = baselines[baselines["route"] == "Route1"].iloc[0]
        self.assertAlmostEqual(r1["pct_vs_similar_routes"], -50.0, places=1)
        self.assertIn("-50.0% vs similar-length routes this week", r1["vs_similar_routes"])


if __name__ == "__main__":
    unittest.main()
