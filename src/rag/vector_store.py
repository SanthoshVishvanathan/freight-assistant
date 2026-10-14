"""
Vector store for context notes using TF-IDF vectorization and semantic indexing.
"""

from dataclasses import dataclass
import datetime
import re
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class NoteDocument:
    note_id: str
    date: str
    applies_to: str
    note_text: str
    start_date: datetime.date
    end_date: datetime.date
    affects_cost_positively: bool
    is_applicable_to_dataset: bool
    embedding: Optional[np.ndarray] = None


class ContextNotesVectorStore:
    """
    In-memory vector store indexing real-world disruption notes.
    Combines dense TF-IDF n-gram vector representations with temporal & spatial metadata.
    """

    def __init__(self, notes_csv_path: Optional[str] = None):
        self.documents: List[NoteDocument] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix: Optional[np.ndarray] = None
        if notes_csv_path:
            self.load_from_csv(notes_csv_path)

    def _parse_dates_and_impact(self, date_str: str, applies_to: str, note_text: str) -> tuple:
        """
        Extracts temporal boundary dates and semantic cost impact indicators.
        """
        ref_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        start_date = ref_date
        end_date = ref_date

        text_lower = note_text.lower()

        # Check for multi-day/week windows in text
        # e.g. Feb 24 to Mar 8
        date_range_match = re.search(r"from\s+([a-zA-Z]+)\s+(\d+)\s+to\s+([a-zA-Z]+)\s+(\d+)", note_text, re.IGNORECASE)
        if date_range_match:
            m1, d1, m2, d2 = date_range_match.groups()
            try:
                dt1 = datetime.datetime.strptime(f"{ref_date.year} {m1} {d1}", "%Y %b %d").date()
                dt2 = datetime.datetime.strptime(f"{ref_date.year} {m2} {d2}", "%Y %b %d").date()
                start_date = min(dt1, dt2)
                end_date = max(dt1, dt2)
            except ValueError:
                pass
        elif "for about a week" in text_lower or "festival week" in text_lower or "starting this week" in text_lower:
            end_date = start_date + datetime.timedelta(days=7)

        # Cost impact indicators
        is_applicable = True
        affects_positively = False

        if "not part of this dataset" in text_lower:
            is_applicable = False
            affects_positively = False
        elif "costs were not significantly affected" in text_lower:
            affects_positively = False
        elif "without a rate change" in text_lower:
            affects_positively = False
        elif "no significant disruptions" in text_lower or "remained normal" in text_lower:
            affects_positively = False
        elif "returned to normal" in text_lower or "improved road conditions" in text_lower:
            affects_positively = False
        elif any(term in text_lower for term in [
            "higher trip costs", "temporary surcharge", "costs rose", "pushing up transportation costs", "detours and higher"
        ]):
            affects_positively = True

        return start_date, end_date, affects_positively, is_applicable

    def load_from_csv(self, csv_path: str) -> None:
        """Loads and indexes notes from CSV."""
        df = pd.read_csv(csv_path)
        self.documents = []
        corpus = []

        for _, row in df.iterrows():
            nid = str(row["note_id"]).strip()
            date_str = str(row["date"]).strip()
            applies_to = str(row["applies_to"]).strip()
            text = str(row["note"]).strip()

            start_d, end_d, cost_pos, app_data = self._parse_dates_and_impact(date_str, applies_to, text)

            # Enrich corpus string with metadata for semantic matching
            enriched_text = f"Route: {applies_to}. Date: {date_str}. Event: {text}"
            corpus.append(enriched_text)

            doc = NoteDocument(
                note_id=nid,
                date=date_str,
                applies_to=applies_to,
                note_text=text,
                start_date=start_d,
                end_date=end_d,
                affects_cost_positively=cost_pos,
                is_applicable_to_dataset=app_data,
            )
            self.documents.append(doc)

        if corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
            for i, doc in enumerate(self.documents):
                doc.embedding = self.tfidf_matrix[i]

    def get_document(self, note_id: str) -> Optional[NoteDocument]:
        """Retrieves document by note_id."""
        for doc in self.documents:
            if doc.note_id == note_id:
                return doc
        return None

    def search_semantic(self, query: str, top_k: int = 3) -> List[tuple]:
        """
        Performs semantic vector search against all indexed documents.
        Returns: List of (NoteDocument, similarity_score)
        """
        if self.tfidf_matrix is None or len(self.documents) == 0:
            return []

        query_vec = self.vectorizer.transform([query]).toarray()
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(sims[idx])))
        return results
