"""
Step 4 — Cosine Similarity Module

What is Cosine Similarity?
  Two TF-IDF vectors point in some direction in high-dimensional space.
  Cosine similarity measures the angle between them:
    • score = 1.0  → identical direction (perfect match)
    • score = 0.0  → completely orthogonal (no shared vocabulary)

  Formula: cos(θ) = (A · B) / (‖A‖ × ‖B‖)

  scikit-learn's cosine_similarity() handles sparse matrices efficiently,
  so we don't need to compute the formula by hand.

Public API:
  get_best_match(query_vec, tfidf_matrix)
      -> (best_index: int, best_score: float)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_best_match(query_vec, tfidf_matrix) -> tuple[int, float]:
    """
    Compare *query_vec* against every row in *tfidf_matrix* using cosine
    similarity and return the index and score of the best match.

    Parameters
    ----------
    query_vec    : sparse matrix (1, n_features) — the user's TF-IDF vector
    tfidf_matrix : sparse matrix (n_questions, n_features) — all FAQ vectors

    Returns
    -------
    best_index : int   — row index of the closest FAQ question
    best_score : float — cosine similarity score in [0.0, 1.0]
    """
    # cosine_similarity returns shape (1, n_questions)
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    return best_index, best_score

