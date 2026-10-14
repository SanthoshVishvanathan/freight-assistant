"""
Precision baseline calculation module.

Defines exact numeric baselines according to grading specification:
1. vs. own history: Trailing 8-week rolling average cost per tonne-km, using only weeks
   strictly before the current week (no look-ahead). If fewer than 8 prior weeks exist,
   use all prior weeks available without padding or extrapolation.
2. vs. similar routes: Average cost per tonne-km in that same week across all other routes
   sharing the same route_type (Short / Medium / Long), excluding the route itself.
"""

from typing import Tuple, Optional
import pandas as pd


class BaselineCalculator:
    """Computes historical and peer baselines for aggregated weekly route metrics."""

    @staticmethod
    def format_percentage(value: Optional[float], suffix: str) -> str:
        """
        Formats percentage difference with sign and 1 decimal place.
        e.g. +35.5% vs this route's past average
        """
        if value is None or pd.isna(value):
            return f"N/A {suffix}"
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.1f}% {suffix}"

    @classmethod
    def compute_for_row(
        cls,
        grouped_df: pd.DataFrame,
        route: str,
        week_of: str,
        route_type: str,
        current_cost: float,
        rolling_window_weeks: int = 8,
    ) -> Tuple[Optional[float], Optional[float], str, str, int]:
        """
        Calculates baseline percentages and formatted strings for a single route-week.

        Returns:
            (pct_own, pct_peers, str_own, str_peers, prior_weeks_count)
        """
        # 1. Own history: strictly prior weeks, up to rolling_window_weeks (tail)
        prior_weeks = grouped_df[
            (grouped_df["route"] == route) & (grouped_df["week_of"] < week_of)
        ]
        trailing_prior = prior_weeks.tail(rolling_window_weeks)
        prior_count = len(trailing_prior)

        if prior_count > 0:
            avg_prior = trailing_prior["cost_per_tonne_km"].mean()
            pct_own = ((current_cost - avg_prior) / avg_prior) * 100.0
            str_own = cls.format_percentage(pct_own, "vs this route's past average")
        else:
            pct_own = None
            str_own = "N/A (first week observed)"

        # 2. Similar routes: same week, same route_type, excluding self
        peer_weeks = grouped_df[
            (grouped_df["week_of"] == week_of)
            & (grouped_df["route_type"] == route_type)
            & (grouped_df["route"] != route)
        ]

        if len(peer_weeks) > 0:
            avg_peers = peer_weeks["cost_per_tonne_km"].mean()
            pct_peers = ((current_cost - avg_peers) / avg_peers) * 100.0
            str_peers = cls.format_percentage(pct_peers, "vs similar-length routes this week")
        else:
            pct_peers = None
            str_peers = "N/A (no peer routes in length category)"

        return pct_own, pct_peers, str_own, str_peers, prior_count

    @classmethod
    def compute_all(
        cls,
        grouped_df: pd.DataFrame,
        rolling_window_weeks: int = 8,
    ) -> pd.DataFrame:
        """
        Calculates historical and peer baselines across the entire aggregated dataset.
        """
        records = []
        for _, row in grouped_df.iterrows():
            route = row["route"]
            week_of = row["week_of"]
            route_type = row["route_type"]
            current_cost = row["cost_per_tonne_km"]

            pct_own, pct_peers, str_own, str_peers, prior_count = cls.compute_for_row(
                grouped_df=grouped_df,
                route=route,
                week_of=week_of,
                route_type=route_type,
                current_cost=current_cost,
                rolling_window_weeks=rolling_window_weeks,
            )

            orig = row.get("origin") or (route.split("-")[0] if "-" in str(route) else str(route))
            dest = row.get("destination") or (route.split("-")[1] if "-" in str(route) else "")

            records.append({
                "route": route,
                "origin": orig,
                "destination": dest,
                "route_type": route_type,
                "week_of": week_of,
                "cost_per_tonne_km": round(current_cost, 2),
                "cost_per_tonne_km_raw": current_cost,
                "prior_weeks_available": prior_count,
                "pct_vs_own_history": pct_own,
                "pct_vs_similar_routes": pct_peers,
                "vs_own_history": str_own,
                "vs_similar_routes": str_peers,
                "total_freight_cost": row.get("total_freight_cost", 0.0),
                "total_tonne_km": row.get("total_tonne_km", 0.0),
                "shipment_count": row.get("shipment_count", 0),
            })

        return pd.DataFrame(records)


def compute_comparison_baselines(
    grouped_df: pd.DataFrame,
    rolling_window_weeks: int = 8,
) -> pd.DataFrame:
    """Convenience functional wrapper for BaselineCalculator."""
    return BaselineCalculator.compute_all(grouped_df, rolling_window_weeks=rolling_window_weeks)
