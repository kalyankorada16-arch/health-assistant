"""A small, dependency-free vector index: TF-IDF vectors compared by cosine similarity.

Chosen so the app deploys with no heavy ML packages. The Retriever interface
(`search`) is deliberately small so an embedding model can replace it later.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "am", "i", "my", "me", "it", "its", "this",
    "that", "these", "those", "have", "has", "had", "do", "does", "did", "can", "could",
    "should", "would", "will", "may", "might", "you", "your", "we", "they", "he", "she",
    "from", "by", "as", "so", "than", "then", "very", "also", "about", "what", "how",
    "when", "which", "who", "get", "got", "feel", "feeling", "since", "some", "any",
}


def _stem(word):
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    words = re.findall(r"[a-z]+", (text or "").lower())
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


@dataclass(frozen=True)
class Hit:
    chunk: object
    score: float


class Retriever:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        token_lists = [tokenize(f"{c.doc.title} {c.text}") for c in self.chunks]
        n = len(token_lists)
        df = Counter()
        for tokens in token_lists:
            df.update(set(tokens))
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = [self._vector(tokens) for tokens in token_lists]

    def _vector(self, tokens):
        counts = Counter(t for t in tokens if t in self.idf)
        vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm else {}

    def search(self, query, k=4, min_score=0.12):
        """Return up to k Hits at or above min_score, best first. Empty list = not covered."""
        terms = set(tokenize(query))
        q = self._vector(list(terms))
        if not q or not self.chunks:
            return []
        # Require more than one shared informative word when the question has several,
        # so a single generic word (e.g. "best") cannot make an unrelated chunk "support" it.
        needed = min(2, len(terms))  # counts words the library has never seen too
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            shared = sum(1 for t in q if t in vec)
            if score >= min_score and shared >= needed:
                scored.append(Hit(chunk, round(score, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
