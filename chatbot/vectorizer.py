"""
Step 3 — TF-IDF Vectorization Module

What is TF-IDF?
  TF  (Term Frequency)     = how often a word appears in a document
  IDF (Inverse Doc Freq.)  = how rare the word is across all documents
  TF-IDF score = TF × IDF  → high score means the word is important *and* rare

We fit TfidfVectorizer on the preprocessed FAQ questions so every question
becomes a numeric vector. When the user asks something, we transform their
preprocessed input into the same vector space and compare with cosine similarity.

Public API:
  build_tfidf(preprocessed_questions: list[str])
      -> (TfidfVectorizer, scipy.sparse matrix)

  transform_query(vectorizer, query: str)
      -> scipy.sparse matrix  (1 × n_features)
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(preprocessed_questions: list[str]):
    """
    Fit a TF-IDF vectorizer on *preprocessed_questions* and return both
    the fitted vectorizer and the resulting document-term matrix.

    Parameters
    ----------
    preprocessed_questions : list[str]
        Each element is an already-preprocessed FAQ question string.

    Returns
    -------
    vectorizer : TfidfVectorizer  (fitted)
    tfidf_matrix : sparse matrix of shape (n_questions, n_features)
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)
    return vectorizer, tfidf_matrix


def transform_query(vectorizer: TfidfVectorizer, query: str):
    """
    Transform a single preprocessed *query* string into the same
    TF-IDF vector space as the FAQ matrix.

    Parameters
    ----------
    vectorizer : TfidfVectorizer  — the already-fitted vectorizer
    query      : str              — preprocessed user question

    Returns
    -------
    sparse matrix of shape (1, n_features)
    """
    return vectorizer.transform([query])

