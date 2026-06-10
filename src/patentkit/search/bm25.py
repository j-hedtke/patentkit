"""In-memory BM25 keyword store — the zero-dependency default backend.

Implements Okapi BM25 over the canonical patent fields with the full
:class:`~patentkit.search.base.SearchQuery` parameter set (required/excluded
tokens, minimum-match, art classes, date cutoffs, allow/deny lists) and
passage highlighting. Suitable for corpora up to ~100k documents; use the
Elasticsearch backend beyond that.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Optional

from patentkit.models import Patent, PatentNumber
from patentkit.search.base import (
    Passage,
    SearchQuery,
    SearchResult,
    apply_metadata_filters,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_FIELD_BOOSTS = {"title": 2.0, "abstract": 1.2, "claims": 1.5, "specification": 1.0}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """Okapi BM25 (k1=1.5, b=0.75) with per-field boosts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._patents: dict[str, Patent] = {}
        self._doc_tokens: dict[str, dict[str, Counter]] = {}  # key -> field -> term counts
        self._doc_len: dict[str, int] = {}
        self._df: Counter = Counter()  # term -> document frequency
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._patents)

    def index(self, patents: Iterable[Patent]) -> int:
        count = 0
        for patent in patents:
            key = str(patent.patent_number)
            if key in self._patents:
                self._remove(key)
            field_tokens = {
                "title": Counter(tokenize(patent.title or "")),
                "abstract": Counter(tokenize(patent.abstract or "")),
                "claims": Counter(tokenize("\n".join(c.text for c in patent.claims))),
                "specification": Counter(tokenize(patent.specification or "")),
            }
            doc_len = sum(sum(c.values()) for c in field_tokens.values())
            self._patents[key] = patent
            self._doc_tokens[key] = field_tokens
            self._doc_len[key] = doc_len
            self._total_len += doc_len
            for term in set().union(*field_tokens.values()):
                self._df[term] += 1
            count += 1
        return count

    def _remove(self, key: str) -> None:
        for term in set().union(*self._doc_tokens[key].values()):
            self._df[term] -= 1
        self._total_len -= self._doc_len[key]
        del self._patents[key], self._doc_tokens[key], self._doc_len[key]

    def get(self, number: PatentNumber) -> Optional[Patent]:
        return self._patents.get(str(number)) or next(
            (p for p in self._patents.values() if p.patent_number.equivalent(number)), None
        )

    def all_patents(self) -> list[Patent]:
        return list(self._patents.values())

    def _idf(self, term: str) -> float:
        n, df = len(self._patents), self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _term_score(self, key: str, term: str, fields: list[str]) -> float:
        if not self._patents:
            return 0.0
        avg_len = self._total_len / len(self._patents)
        tf = sum(
            self._doc_tokens[key].get(f, Counter()).get(term, 0) * _FIELD_BOOSTS.get(f, 1.0)
            for f in fields
        )
        if tf == 0:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * self._doc_len[key] / max(avg_len, 1))
        return self._idf(term) * tf * (self.k1 + 1) / (tf + norm)

    def search(self, query: SearchQuery) -> list[SearchResult]:
        terms = [t for kw in query.keywords for t in tokenize(kw)]
        text_terms = tokenize(query.text or "")
        required = [tokenize(kw) for kw in query.required_keywords]
        all_terms = list(dict.fromkeys(terms + text_terms))
        minimum_match = query.effective_minimum_match() if query.keywords else 0

        results = []
        for key, patent in self._patents.items():
            if not apply_metadata_filters(patent, query):
                continue
            doc_terms = set().union(*self._doc_tokens[key].values())
            if required and not all(
                all(tok in doc_terms for tok in group) for group in required
            ):
                continue
            matched_keywords = sum(
                1 for kw in query.keywords if all(t in doc_terms for t in tokenize(kw))
            )
            if query.keywords and matched_keywords < minimum_match:
                continue
            score = sum(self._term_score(key, term, query.fields) for term in all_terms)
            if score <= 0 and all_terms:
                continue
            results.append(SearchResult(
                patent_number=patent.patent_number,
                score=score,
                patent=patent,
                passages=self._highlight(patent, all_terms),
                explanation=f"bm25 matched {matched_keywords}/{len(query.keywords)} keywords",
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.limit]

    def _highlight(self, patent: Patent, terms: list[str], window: int = 240,
                   max_passages: int = 3) -> list[Passage]:
        """Pick the spec/claim windows densest in query terms."""
        term_set = set(terms)
        candidates: list[Passage] = []
        sources = [("claims", "\n".join(c.text for c in patent.claims)),
                   ("specification", patent.specification or ""),
                   ("abstract", patent.abstract or "")]
        for field_name, text in sources:
            if not text:
                continue
            for start in range(0, len(text), window):
                chunk = text[start:start + window]
                hits = sum(1 for t in tokenize(chunk) if t in term_set)
                if hits:
                    candidates.append(Passage(
                        text=chunk.strip(), field=field_name, score=float(hits),
                        start=start, end=start + len(chunk),
                    ))
        candidates.sort(key=lambda p: p.score, reverse=True)
        return candidates[:max_passages]
