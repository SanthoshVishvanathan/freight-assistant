"""
Anomaly detection engine.

Identifies route-weeks where shipping costs are rising and look out of the ordinary
compared to own trailing history OR to similar routes.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class AnomalyCriteria:
    """Configurable thresholds for flagging route anomalies."""
    own_history_threshold_pct: float = 20.0
    peer_routes_threshold_pct: float = 20.0
    require_positive_rise: bool = True  # Cost must be higher than historical baseline


@dataclass
class RouteAnomalyCandidate:
    route: str
    week_of: str
    route_type: str
    cost_per_tonne_km: float
    pct_vs_own_history: Optional[float]
    pct_vs_similar_routes: Optional[float]
    vs_own_history_str: str
    vs_similar_routes_str: str
    trigger_reason: str


class AnomalyDetector:
    """Evaluates route-week metrics against anomaly criteria."""

    def __init__(self, criteria: Optional[AnomalyCriteria] = None):
        self.criteria = criteria or AnomalyCriteria()

    def evaluate_row(self, row: pd.Series) -> Optional[RouteAnomalyCandidate]:
        """
        Determines if a route-week record should be flagged as an anomaly candidate.
        """
        pct_own = row.get("pct_vs_own_history")
        pct_peer = row.get("pct_vs_similar_routes")

        triggers = []
        is_candidate = False

        # Condition 1: Significant rise vs own past average
        if pct_own is not None and pct_own >= self.criteria.own_history_threshold_pct:
            triggers.append(f"Own history spike (+{pct_own:.1f}% >= +{self.criteria.own_history_threshold_pct:.1f}%)")
            is_candidate = True

        # Condition 2: Significant divergence vs similar routes in same week
        if pct_peer is not None and pct_peer >= self.criteria.peer_routes_threshold_pct:
            # If required, ensure cost is not falling compared to own history
            if not self.criteria.require_positive_rise or (pct_own is None or pct_own > 0.0):
                triggers.append(f"Peer corridor deviation (+{pct_peer:.1f}% >= +{self.criteria.peer_routes_threshold_pct:.1f}%)")
                is_candidate = True

        if not is_candidate:
            return None

        return RouteAnomalyCandidate(
            route=row["route"],
            week_of=row["week_of"],
            route_type=row["route_type"],
            cost_per_tonne_km=row["cost_per_tonne_km"],
            pct_vs_own_history=pct_own,
            pct_vs_similar_routes=pct_peer,
            vs_own_history_str=row["vs_own_history"],
            vs_similar_routes_str=row["vs_similar_routes"],
            trigger_reason=" & ".join(triggers)
        )

    def detect(self, baselines_df: pd.DataFrame) -> List[RouteAnomalyCandidate]:
        """
        Scans all records and returns the list of anomaly candidates requiring RAG inspection.
        """
        candidates = []
        for _, row in baselines_df.iterrows():
            cand = self.evaluate_row(row)
            if cand is not None:
                candidates.append(cand)
        return candidates
