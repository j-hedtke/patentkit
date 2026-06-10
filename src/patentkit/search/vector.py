"""Embedding providers and the in-memory vector/RAG store.

Embedding providers follow the bring-your-own-key pattern; the in-memory
store is pure Python (uses numpy when available) and chunks specifications
with overlap, mirroring the production strategy (1500-token chunks, 20%
overlap) at word granularity.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from patentkit.config import resolve_key
from patentkit.models import Patent, PatentNumber, SpecChunk
from patentkit.search.base import Passage, SearchQuery, SearchResult, apply_metadata_filters

CHUNK_WORDS = 700  # ~1500 tokens
CHUNK_OVERLAP = 0.2


def chunk_text(text: str, max_words: int = CHUNK_WORDS, overlap: float = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, int(max_words * (1 - overlap)))
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)]


def cosine(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np
        va, vb = np.asarray(a), np.asarray(b)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(va @ vb / denom) if denom else 0.0
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / denom if denom else 0.0


class OpenAIEmbeddings:
    """text-embedding-3 family (requires ``patentkit[openai]``)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None,
                 dimensions: int | None = None):
        self.model_name = model
        self.api_key = api_key
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        import openai
        client = openai.OpenAI(api_key=resolve_key("OPENAI_API_KEY", self.api_key))
        kwargs = {"dimensions": self.dimensions} if self.dimensions else {}
        out: list[list[float]] = []
        for batch_start in range(0, len(texts), 128):
            batch = texts[batch_start:batch_start + 128]
            response = client.embeddings.create(model=self.model_name, input=batch, **kwargs)
            out += [item.embedding for item in response.data]
        return out


class VoyageEmbeddings:
    """Voyage AI embeddings — Anthropic's recommended embedding partner."""

    def __init__(self, model: str = "voyage-3", api_key: str | None = None):
        self.model_name = model
        self.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        key = resolve_key("VOYAGE_API_KEY", self.api_key)
        out: list[list[float]] = []
        for batch_start in range(0, len(texts), 128):
            batch = texts[batch_start:batch_start + 128]
            response = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": self.model_name, "input": batch},
                timeout=120,
            )
            response.raise_for_status()
            out += [item["embedding"] for item in response.json()["data"]]
        return out


class HashingEmbeddings:
    """Deterministic local embeddings (token hashing) — for tests/offline demos.

    Not semantically meaningful beyond vocabulary overlap; never use for
    production relevance.
    """

    def __init__(self, dimensions: int = 256):
        self.model_name = f"hashing-{dimensions}"
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for token in text.lower().split():
                vec[hash(token) % self.dimensions] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


class InMemoryVectorStore:
    """Chunk + embed + brute-force cosine search. Default RAG store."""

    def __init__(self, embeddings, chunk_words: int = CHUNK_WORDS):
        self.embeddings = embeddings
        self.chunk_words = chunk_words
        self._patents: dict[str, Patent] = {}
        self._chunks: list[tuple[str, SpecChunk]] = []  # (patent key, chunk)

    def index(self, patents: Iterable[Patent]) -> int:
        count = 0
        for patent in patents:
            key = str(patent.patent_number)
            self._patents[key] = patent
            self._chunks = [(k, c) for k, c in self._chunks if k != key]
            texts = chunk_text(patent.text_for_search(), self.chunk_words)
            if not texts:
                continue
            vectors = self.embeddings.embed(texts)
            for i, (text, vec) in enumerate(zip(texts, vectors)):
                self._chunks.append((key, SpecChunk(
                    chunk_number=i, text=text, embedding=vec,
                    embedding_model=self.embeddings.model_name,
                )))
            count += 1
        return count

    def get(self, number: PatentNumber) -> Optional[Patent]:
        return self._patents.get(str(number)) or next(
            (p for p in self._patents.values() if p.patent_number.equivalent(number)), None
        )

    def search_text(self, text: str, *, limit: int = 50,
                    query: SearchQuery | None = None) -> list[SearchResult]:
        if not self._chunks:
            return []
        [query_vec] = self.embeddings.embed([text])
        best: dict[str, tuple[float, SpecChunk]] = {}
        for key, chunk in self._chunks:
            patent = self._patents[key]
            if query and not apply_metadata_filters(patent, query):
                continue
            score = cosine(query_vec, chunk.embedding or [])
            if key not in best or score > best[key][0]:
                best[key] = (score, chunk)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
        return [
            SearchResult(
                patent_number=self._patents[key].patent_number,
                score=score,
                patent=self._patents[key],
                passages=[Passage(text=chunk.text[:400], field="specification", score=score)],
                explanation=f"cosine similarity ({self.embeddings.model_name})",
            )
            for key, (score, chunk) in ranked
        ]
