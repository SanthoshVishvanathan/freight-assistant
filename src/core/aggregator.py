"""
Aggregates individual shipment records into weekly route-level metrics.

Rules:
- Route is defined by origin-destination (e.g. Delhi-Jaipur).
- Route type is preserved as provided (Short / Medium / Long).
- Weeks run Monday to Sunday; week_of is that week's Monday date (YYYY-MM-DD).
- Cost per tonne per km = total freight cost / (quantity carried in tonnes x distance travelled in km).
"""

from dataclasses import dataclass
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


@dataclass(frozen=True)
class WeeklyRouteMetric:
    route: str
    origin: str
    destination: str
    route_type: str
    week_of: str
    total_freight_cost: float
    total_tonne_km: float
    cost_per_tonne_km: float
    shipment_count: int


class ShipmentAggregator:
    """Handles parsing, validation, and weekly aggregation of shipment records."""

    REQUIRED_COLUMNS = [
        "shipment_id", "origin", "destination", "route_type",
        "material", "quantity_tonnes", "distance_km",
        "freight_cost_inr", "shipment_date"
    ]

    @staticmethod
    def calculate_week_of(date_val: datetime.date) -> str:
        """Returns the Monday date string (YYYY-MM-DD) for any given date."""
        monday = date_val - datetime.timedelta(days=date_val.weekday())
        return monday.strftime("%Y-%m-%d")

    @classmethod
    def aggregate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and groups shipment records into weekly route-level aggregates.
        
        Args:
            df: DataFrame containing raw shipment records.
            
        Returns:
            pd.DataFrame sorted by route and week_of with weekly aggregated metrics.
        """
        missing = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required shipment columns: {missing}")

        data = df.copy()
        data["shipment_date"] = pd.to_datetime(data["shipment_date"])
        data["week_of"] = data["shipment_date"].dt.date.apply(cls.calculate_week_of)
        data["route"] = data["origin"].str.strip() + "-" + data["destination"].str.strip()
        data["tonne_km"] = data["quantity_tonnes"] * data["distance_km"]

        grouped = (
            data.groupby(["route", "origin", "destination", "route_type", "week_of"], as_index=False)
            .agg(
                total_freight_cost=("freight_cost_inr", "sum"),
                total_tonne_km=("tonne_km", "sum"),
                shipment_count=("shipment_id", "count")
            )
        )

        grouped["cost_per_tonne_km"] = (
            grouped["total_freight_cost"] / grouped["total_tonne_km"]
        )

        # Sort chronologically by route and week
        grouped = grouped.sort_values(["route", "week_of"]).reset_index(drop=True)
        return grouped


def aggregate_weekly_shipments(df_or_path: Any) -> pd.DataFrame:
    """Convenience functional wrapper for ShipmentAggregator."""
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path
    return ShipmentAggregator.aggregate(df)
