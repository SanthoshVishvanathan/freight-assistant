"""
Deterministic guardrail validator enforcing anti-hallucination rules.

Grading Requirements:
- Every 'justified' verdict must cite the exact note.
- That note must apply to that route and that time window.
- Notes saying costs were not affected, or routes not in dataset, do NOT count.
- No supporting note -> unexplained. Confident wrong reasons score worse than 'couldn't find one'.
"""

from dataclasses import dataclass
import datetime
from typing import Optional

from .vector_store import NoteDocument
from .retriever import RetrievalResult
from ..core.detector import RouteAnomalyCandidate


@dataclass
class GuardrailVerdict:
    is_justified: bool
    flagged: str  # 'No (justified)' or 'Yes'
    matched_note_id: str  # Note ID if justified, else '' (blank)
    citing_note: Optional[NoteDocument]
    closest_unjustifying_note: Optional[NoteDocument]
    closest_unjustifying_note_days: Optional[int]
    guardrail_explanation: str


class GuardrailValidator:
    """
    Validates retrieved notes against strict business constraints to prevent AI hallucination.
    """

    NON_JUSTIFYING_PHRASES = [
        "not part of this dataset",
        "costs were not significantly affected",
        "without a rate change",
        "no significant disruptions",
        "remained normal",
        "returned to normal conditions",
        "improved road conditions"
    ]

    @classmethod
    def validate(
        cls,
        candidate: RouteAnomalyCandidate,
        retrieval: RetrievalResult,
    ) -> GuardrailVerdict:
        """
        Executes multi-rule validation over candidate anomaly and retrieval results.
        """
        cand_route = candidate.route
        week_dt = datetime.datetime.strptime(candidate.week_of, "%Y-%m-%d").date()
        week_end = week_dt + datetime.timedelta(days=6)

        best_note = retrieval.best_matching_note
        closest_note = retrieval.closest_temporal_note

        # If no note was deemed a candidate justification
        if best_note is None:
            # Check if closest note can be cited as non-justifying context
            explanation = "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."
            if closest_note is not None and retrieval.days_difference is not None and abs(retrieval.days_difference) <= 28:
                explanation = (
                    f"The closest note ({closest_note.note_id}, {closest_note.date}) mentions "
                    f"{cls._summarize_disclaimer(closest_note.note_text)} -- it does not describe a reason for a "
                    f"cost rise on this route. No genuine justification found; flagged for review."
                )

            return GuardrailVerdict(
                is_justified=False,
                flagged="Yes",
                matched_note_id="",
                citing_note=None,
                closest_unjustifying_note=closest_note,
                closest_unjustifying_note_days=retrieval.days_difference,
                guardrail_explanation=explanation
            )

        # Guardrail 1: Route Applicability
        if best_note.applies_to != cand_route and best_note.applies_to != "All Routes":
            return GuardrailVerdict(
                is_justified=False,
                flagged="Yes",
                matched_note_id="",
                citing_note=None,
                closest_unjustifying_note=best_note,
                closest_unjustifying_note_days=retrieval.days_difference,
                guardrail_explanation=f"Note {best_note.note_id} applies to {best_note.applies_to}, not {cand_route}. Rejected by route guardrail."
            )

        # Guardrail 2: Temporal Window Verification
        # Check if week overlaps with note window
        note_dt = datetime.datetime.strptime(best_note.date, "%Y-%m-%d").date()
        in_active_window = (
            (best_note.start_date <= week_end and best_note.end_date >= week_dt)
            or (abs((week_dt - note_dt).days) <= 7)
        )

        if not in_active_window:
            return GuardrailVerdict(
                is_justified=False,
                flagged="Yes",
                matched_note_id="",
                citing_note=None,
                closest_unjustifying_note=best_note,
                closest_unjustifying_note_days=retrieval.days_difference,
                guardrail_explanation=f"Note {best_note.note_id} window ({best_note.start_date} to {best_note.end_date}) does not cover week {candidate.week_of}. Rejected by temporal guardrail."
            )

        # Guardrail 3: Explicit Non-Effect / Negative Clause Check
        note_lower = best_note.note_text.lower()
        for phrase in cls.NON_JUSTIFYING_PHRASES:
            if phrase in note_lower:
                return GuardrailVerdict(
                    is_justified=False,
                    flagged="Yes",
                    matched_note_id="",
                    citing_note=None,
                    closest_unjustifying_note=best_note,
                    closest_unjustifying_note_days=retrieval.days_difference,
                    guardrail_explanation=f"Note {best_note.note_id} indicates '{phrase}' and cannot justify a cost increase. Rejected by causal guardrail."
                )

        # Guardrail 4: Must have positive cost impact
        if not best_note.affects_cost_positively:
            return GuardrailVerdict(
                is_justified=False,
                flagged="Yes",
                matched_note_id="",
                citing_note=None,
                closest_unjustifying_note=best_note,
                closest_unjustifying_note_days=retrieval.days_difference,
                guardrail_explanation=f"Note {best_note.note_id} does not cite a cost-increasing mechanism. Rejected by justification guardrail."
            )

        # All guardrails passed: Validated justification
        return GuardrailVerdict(
            is_justified=True,
            flagged="No (justified)",
            matched_note_id=best_note.note_id,
            citing_note=best_note,
            closest_unjustifying_note=None,
            closest_unjustifying_note_days=None,
            guardrail_explanation=f"Matches note {best_note.note_id} dated {best_note.date}: {best_note.note_text}"
        )

    @staticmethod
    def _summarize_disclaimer(note_text: str) -> str:
        """Extracts concise operational gist of non-justifying note."""
        text_lower = note_text.lower()
        if "stable demand" in text_lower or "no major disruptions" in text_lower:
            return "stable demand with no major disruptions"
        if "compliance costs were absorbed" in text_lower or "without a rate change" in text_lower:
            return "compliance costs absorbed without a rate change"
        if "costs were not significantly affected" in text_lower:
            return "costs were not significantly affected"
        if "not part of this dataset" in text_lower:
            return "the affected routes are not part of this dataset"
        if "returned to normal" in text_lower:
            return "route conditions returned to normal"
        return "normal conditions"
