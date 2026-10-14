"""
Interactive Query Engine answering natural language questions using assistant findings.
"""

import datetime
import re
from typing import List, Dict, Any, Optional
import pandas as pd

from ..reasoning.engine import RouteExplanation


class ShippingCostAssistant:
    """
    Stretch Goal: Conversational interface answering operational questions
    grounded directly in the assistant's computed baselines and notes.
    """

    MONTHS = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    def __init__(
        self,
        explanations: List[RouteExplanation],
        all_baselines_df: Optional[pd.DataFrame] = None,
        notes_df: Optional[pd.DataFrame] = None,
    ):
        self.explanations = explanations
        self.baselines_df = all_baselines_df
        self.notes_df = notes_df
        self.known_routes = []
        if all_baselines_df is not None:
            self.known_routes = list(all_baselines_df["route"].unique())

    def _extract_route(self, question: str) -> Optional[str]:
        """Detects mentioned route corridor in question."""
        q_lower = question.lower().replace(" ", "-")
        for r in self.known_routes:
            if r.lower() in q_lower:
                return r
        # Try checking origin and destination separately
        for r in self.known_routes:
            orig, dest = r.split("-")
            if orig.lower() in question.lower() and dest.lower() in question.lower():
                return r
        return None

    def _extract_month_and_year(self, question: str) -> tuple:
        """Extracts month number and year if mentioned."""
        q_lower = question.lower()
        target_month = None
        for m_name, m_num in self.MONTHS.items():
            if re.search(r"\b" + m_name + r"\b", q_lower):
                target_month = m_num
                break

        year_match = re.search(r"\b(2024|2025)\b", question)
        target_year = int(year_match.group(1)) if year_match else None

        return target_month, target_year

    def answer(self, question: str) -> str:
        """
        Synthesizes a fact-based answer strictly grounded in assistant results.
        """
        route = self._extract_route(question)
        month, year = self._extract_month_and_year(question)

        # Overview query
        if "all" in question.lower() and ("flag" in question.lower() or "summary" in question.lower() or "anomal" in question.lower()):
            justified_count = sum(1 for e in self.explanations if "justified" in e.flagged.lower())
            unexplained_count = sum(1 for e in self.explanations if e.flagged == "Yes")
            lines = [
                f"Assistant Findings Summary:",
                f"- Total anomalous route-weeks identified: {len(self.explanations)}",
                f"- Legitimate / Justified cost rises: {justified_count}",
                f"- Unexplained spikes flagged for human review: {unexplained_count}",
                "\nKey Highlights:",
            ]
            for exp in self.explanations[:5]:
                lines.append(f"• {exp.route} ({exp.week_of}): {exp.cost_per_tonne_km:.2f} INR/tonne-km | Flagged: {exp.flagged} | {exp.reason}")
            return "\n".join(lines)

        # Route-specific query
        matching_exps = []
        for exp in self.explanations:
            if route and exp.route != route:
                continue
            if month or year:
                exp_dt = datetime.datetime.strptime(exp.week_of, "%Y-%m-%d").date()
                if month and exp_dt.month != month:
                    continue
                if year and exp_dt.year != year:
                    continue
            matching_exps.append(exp)

        if matching_exps:
            responses = []
            for exp in matching_exps:
                verdict_str = "justified by operational events" if "justified" in exp.flagged.lower() else "unexplained and flagged for review"
                responses.append(
                    f"During the week of {exp.week_of}, {exp.route} cost was {exp.cost_per_tonne_km:.2f} INR/tonne-km "
                    f"({exp.vs_own_history}, {exp.vs_similar_routes}).\n"
                    f"Verdict: {exp.flagged} ({verdict_str}).\n"
                    f"Explanation: {exp.reason}"
                )
            return "\n\n".join(responses)

        # If no explanation record matched, check baseline data
        if route and self.baselines_df is not None:
            sub = self.baselines_df[self.baselines_df["route"] == route]
            if month or year:
                sub_records = []
                for _, r in sub.iterrows():
                    d = datetime.datetime.strptime(r["week_of"], "%Y-%m-%d").date()
                    if month and d.month != month:
                        continue
                    if year and d.year != year:
                        continue
                    sub_records.append(r)
                if sub_records:
                    avg_c = sum(r["cost_per_tonne_km"] for r in sub_records) / len(sub_records)
                    return (
                        f"For {route} in the requested timeframe, shipping costs averaged {avg_c:.2f} INR/tonne-km "
                        f"with no abnormal cost surges exceeding the 20% anomaly threshold. All operations appeared within normal variation."
                    )

        return (
            "I could not find an anomalous cost surge matching your specific query. "
            "Please specify a route (e.g., 'Ahmedabad-Mumbai', 'Delhi-Jaipur', 'Chennai-Bangalore', 'Mumbai-Pune') "
            "or a date period (e.g., 'March 2025', 'November 2024')."
        )
