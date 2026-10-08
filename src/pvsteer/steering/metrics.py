"""Noisy generation metrics. Lexicon sentiment is not ground truth."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pvsteer.constants import POSITIVE_CATEGORIES

POSITIVE_LEXICON = {
    "satisfied",
    "satisfaction",
    "calm",
    "content",
    "contentment",
    "curious",
    "curiosity",
    "grateful",
    "pleasant",
    "elegant",
    "success",
    "warm",
    "interest",
    "relieved",
    "relief",
    "fulfilled",
    "fulfillment",
    "welcome",
    "beautiful",
    "enjoy",
    "glad",
}
NEGATIVE_LEXICON = {
    "annoyed",
    "displeased",
    "threat",
    "danger",
    "risk",
    "indifferent",
    "numb",
    "dislike",
    "tense",
}
HARMFUL_KEYWORDS = {"kill", "murder", "bomb", "hate", "attack"}
_TOKEN = re.compile(r"[A-Za-z']+")
_FIRST = re.compile(r"\b(I|me|my|mine)\b")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def lexicon_rate(text: str, lexicon: set[str]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in lexicon)
    return hits / len(tokens)


def sentiment_score(text: str) -> float:
    """Difference of lexicon rates. This is a noisy proxy, not a label."""
    return lexicon_rate(text, POSITIVE_LEXICON) - lexicon_rate(text, NEGATIVE_LEXICON)


def first_person_rate(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(_FIRST.findall(text)) / len(tokens)


def repetition_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    value = 0.0
    for count in counts.values():
        probability = count / total
        value -= probability * math.log(probability + 1e-12)
    return float(value)


def distinct_n(tokens: list[str], n: int = 2) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def harmful_flag(text: str) -> bool:
    return any(token in HARMFUL_KEYWORDS for token in tokenize(text))


def bow_cosine(left: str, right: str) -> float | None:
    if not left.strip() or not right.strip():
        return None
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform([left, right])
    except ValueError:
        return None
    left_row = matrix[0].toarray().ravel()
    right_row = matrix[1].toarray().ravel()
    denom = np.linalg.norm(left_row) * np.linalg.norm(right_row)
    if denom == 0:
        return None
    return float(np.dot(left_row, right_row) / denom)


class CategoryProbe:
    """Local TF-IDF classifier trained on the synthetic dataset. Not ground truth."""

    def __init__(self, records: list[dict]):
        self.vectorizer = TfidfVectorizer(min_df=1)
        texts = [row["text"] for row in records]
        labels = [row["category"] for row in records]
        features = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(max_iter=400)
        self.classifier.fit(features, labels)
        self.classes = list(self.classifier.classes_)

    def positive_probability(self, text: str) -> float | None:
        if not text.strip():
            return None
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        total = 0.0
        for category in POSITIVE_CATEGORIES:
            if category in self.classes:
                total += float(probabilities[self.classes.index(category)])
        return total


def generation_metrics(text: str, baseline_text: str | None, probe: CategoryProbe | None) -> dict:
    tokens = tokenize(text)
    return {
        "positive_lexicon_rate": lexicon_rate(text, POSITIVE_LEXICON),
        "sentiment_score": sentiment_score(text),
        "category_classifier_score": None if probe is None else probe.positive_probability(text),
        "first_person_rate": first_person_rate(text),
        "repetition_rate": repetition_rate(tokens),
        "entropy": entropy(tokens),
        "distinct_2": distinct_n(tokens, 2),
        "harmful_content_flag": harmful_flag(text),
        "similarity_to_baseline": None if baseline_text is None else bow_cosine(text, baseline_text),
        "sentiment_is_ground_truth": False,
    }
