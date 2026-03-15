"""
Steps 5 & 6 — FAQ Chatbot Engine

This module ties together all the other modules:
  1. Loads FAQs from data/faqs.csv using pandas
  2. Preprocesses every FAQ question (Step 2)
  3. Builds the TF-IDF matrix at startup (Step 3)
  4. On each user query:
       a. Preprocesses the query
       b. Transforms it into a TF-IDF vector
       c. Finds the best matching FAQ via cosine similarity (Step 4)
       d. Checks the confidence threshold (Step 6)
       e. Logs unknown questions when below threshold (Step 6)

Public API:
  FAQEngine(csv_path, threshold=0.4)
      .ask(question: str) -> dict  with keys: answer, score, matched_question
"""

import os
import csv
from datetime import datetime

import pandas as pd

from chatbot.preprocess import preprocess
from chatbot.vectorizer import build_tfidf, transform_query
from chatbot.similarity import get_best_match

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV   = os.path.join(_BASE_DIR, "data", "faqs.csv")
UNKNOWN_LOG   = os.path.join(_BASE_DIR, "logs", "unknown_questions.csv")

# ---------------------------------------------------------------------------
# Fallback messages
# ---------------------------------------------------------------------------
FALLBACK_MESSAGE = (
    "I'm sorry, I don't have a confident answer to that question. "
    "Please contact the university admissions office for further assistance."
)


class FAQEngine:
    """
    Core chatbot engine.

    Parameters
    ----------
    csv_path  : str   — path to the FAQs CSV file (columns: question, answer)
    threshold : float — minimum cosine similarity score to return an answer
                        (default 0.4; questions below this are logged)
    """

    def __init__(self, csv_path: str = DEFAULT_CSV, threshold: float = 0.4):
        self.threshold = threshold
        self._load_faqs(csv_path)
        self._build_index()
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_faqs(self, csv_path: str) -> None:
        """Load questions and answers from the CSV into memory."""
        df = pd.read_csv(csv_path)
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError(
                f"CSV at '{csv_path}' must contain 'question' and 'answer' columns."
            )
        self.questions: list[str] = df["question"].tolist()
        self.answers:   list[str] = df["answer"].tolist()

    def _build_index(self) -> None:
        """Preprocess all FAQ questions and build the TF-IDF matrix."""
        self.preprocessed_questions = [preprocess(q) for q in self.questions]
        self.vectorizer, self.tfidf_matrix = build_tfidf(
            self.preprocessed_questions
        )

    def _ensure_log_file(self) -> None:
        """Create the unknown_questions log file with headers if it doesn't exist."""
        os.makedirs(os.path.dirname(UNKNOWN_LOG), exist_ok=True)
        if not os.path.exists(UNKNOWN_LOG) or os.path.getsize(UNKNOWN_LOG) == 0:
            with open(UNKNOWN_LOG, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "question", "best_score"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Find the best FAQ answer for *question*.

        Returns a dict:
          {
            "answer":           str,   # the answer text (or fallback)
            "score":            float, # cosine similarity score
            "matched_question": str,   # the original FAQ question matched
            "confident":        bool,  # True if score >= threshold
          }
        """
        cleaned   = preprocess(question)
        query_vec = transform_query(self.vectorizer, cleaned)
        idx, score = get_best_match(query_vec, self.tfidf_matrix)

        confident = score >= self.threshold

        if confident:
            answer          = self.answers[idx]
            matched_question = self.questions[idx]
        else:
            answer          = FALLBACK_MESSAGE
            matched_question = self.questions[idx]   # still informative
            self._log_unknown(question, score)

        return {
            "answer":           answer,
            "score":            round(score, 4),
            "matched_question": matched_question,
            "confident":        confident,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_unknown(self, question: str, score: float) -> None:
        """Append an unrecognised question to the unknown_questions log."""
        with open(UNKNOWN_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question,
                round(score, 4),
            ])

