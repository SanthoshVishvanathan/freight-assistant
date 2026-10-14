"""
Core domain logic for freight aggregation, rolling baselines, and anomaly detection.
"""

from .aggregator import aggregate_weekly_shipments, ShipmentAggregator
from .baselines import compute_comparison_baselines, BaselineCalculator
from .detector import AnomalyDetector, RouteAnomalyCandidate

__all__ = [
    "aggregate_weekly_shipments",
    "ShipmentAggregator",
    "compute_comparison_baselines",
    "BaselineCalculator",
    "AnomalyDetector",
    "RouteAnomalyCandidate",
]
