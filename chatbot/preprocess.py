"""
Step 2 — NLP Preprocessing Module

Pipeline (applied in order):
  1. Lowercase
  2. Synonym normalization  (tuition→fees, course→program, lecturer→professor)
  3. Punctuation removal    (keep only alphabetic tokens)
  4. Tokenization           (nltk word_tokenize)
  5. Stopword removal       (English stopwords)
  6. Lemmatization          (WordNetLemmatizer)

Public API:
  preprocess(text: str) -> str
      Returns a single space-joined string of cleaned tokens, ready for
      TF-IDF vectorisation.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Synonym dictionary – expand as needed
# ---------------------------------------------------------------------------
SYNONYMS: dict[str, str] = {
    "tuition": "fees",
    "course":  "program",
    "courses": "programs",
    "lecturer": "professor",
    "lecturers": "professors",
    "teacher": "professor",
    "teachers": "professors",
    "uni": "university",
    "college": "university",
    "enroll": "apply",
    "enrollment": "admission",
    "register": "apply",
    "registration": "admission",
    "dorm": "housing",
    "dormitory": "housing",
    "accommodation": "housing",
    "class": "program",
    "classes": "programs",
    "grade": "result",
    "grades": "results",
    "scholarship": "scholarship",   # kept explicit for clarity
}

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def _normalize_synonyms(text: str) -> str:
    """Replace each synonym token with its canonical form."""
    words = text.split()
    return " ".join(SYNONYMS.get(w, w) for w in words)


def preprocess(text: str) -> str:
    """
    Apply the full NLP preprocessing pipeline to *text* and return a
    cleaned, space-joined string of lemmatized tokens.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Synonym normalization (done on whole string before tokenization)
    text = _normalize_synonyms(text)

    # 3. Remove punctuation – keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    # 4. Tokenize
    tokens = word_tokenize(text)

    # 5. Remove stopwords and non-alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and t not in _stop_words]

    # 6. Lemmatize
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

