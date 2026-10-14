"""
Hybrid retriever combining semantic search, metadata filters, and temporal alignment.
"""

from dataclasses import dataclass
import datetime
from typing import List, Optional
import numpy as np

from .vector_store import ContextNotesVectorStore, NoteDocument
from ..core.detector import RouteAnomalyCandidate


@dataclass
class RetrievalResult:
    best_matching_note: Optional[NoteDocument]
    closest_temporal_note: Optional[NoteDocument]
    similarity_score: float
    days_difference: Optional[int]
    is_temporal_window_match: bool
    is_corridor_match: bool
    all_scored_candidates: List[tuple]


class ContextRetriever:
    """
    RAG retriever implementing multi-stage filtering:
    1. Spatial / Corridor Filtering: route match or All Routes.
    2. Temporal Proximity: distance in days and exact event window overlap.
    3. Dense Vector Similarity: TF-IDF semantic cosine similarity.
    """

    def __init__(self, vector_store: ContextNotesVectorStore):
        self.vector_store = vector_store

    def retrieve_for_candidate(
        self,
        candidate: RouteAnomalyCandidate,
        max_temporal_days_window: int = 21,
    ) -> RetrievalResult:
        """
        Retrieves the most relevant context notes for a given route anomaly.
        """
        target_route = candidate.route
        week_dt = datetime.datetime.strptime(candidate.week_of, "%Y-%m-%d").date()
        week_end = week_dt + datetime.timedelta(days=6)

        query = f"High freight shipping cost surge on {target_route} during {candidate.week_of} fuel toll surcharge weather detour disruption"
        semantic_results = self.vector_store.search_semantic(query, top_k=len(self.vector_store.documents))

        scored_candidates = []
        best_justification_note = None
        best_score = -1.0

        closest_note = None
        min_abs_days = 999999

        for doc, sim in semantic_results:
            # 1. Spatial alignment
            is_corridor = (doc.applies_to == target_route or doc.applies_to == "All Routes")
            
            # 2. Temporal distance
            note_date = datetime.datetime.strptime(doc.date, "%Y-%m-%d").date()
            diff_days = (week_dt - note_date).days

            # Track closest note temporally for corridor or all routes
            if is_corridor and abs(diff_days) < min_abs_days:
                min_abs_days = abs(diff_days)
                closest_note = doc

            # Check if week overlaps with note active window
            # Active window: [start_date, end_date] or within 7 days
            in_window = (
                (doc.start_date <= week_end and doc.end_date >= week_dt)
                or (abs(diff_days) <= 7)
            )

            # Composite hybrid score
            hybrid_score = sim
            if is_corridor:
                hybrid_score += 1.0
            if in_window:
                hybrid_score += 1.0
            if doc.affects_cost_positively:
                hybrid_score += 0.5
            if not doc.is_applicable_to_dataset:
                hybrid_score -= 2.0

            scored_candidates.append((doc, hybrid_score, diff_days, in_window, is_corridor))

            if is_corridor and in_window and doc.affects_cost_positively and doc.is_applicable_to_dataset:
                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_justification_note = doc

        # Sort all scored candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        if best_justification_note:
            note_date = datetime.datetime.strptime(best_justification_note.date, "%Y-%m-%d").date()
            days_diff = (week_dt - note_date).days
            return RetrievalResult(
                best_matching_note=best_justification_note,
                closest_temporal_note=closest_note or best_justification_note,
                similarity_score=best_score,
                days_difference=days_diff,
                is_temporal_window_match=True,
                is_corridor_match=True,
                all_scored_candidates=scored_candidates,
            )
        else:
            closest_diff = (week_dt - datetime.datetime.strptime(closest_note.date, "%Y-%m-%d").date()).days if closest_note else None
            return RetrievalResult(
                best_matching_note=None,
                closest_temporal_note=closest_note,
                similarity_score=0.0,
                days_difference=closest_diff,
                is_temporal_window_match=False,
                is_corridor_match=(closest_note is not None),
                all_scored_candidates=scored_candidates,
            )
