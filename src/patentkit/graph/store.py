"""File-backed stores for match pairs (JSONL) and the concept graph (JSON).

Both implement thin abstract backends (:class:`MatchPairBackend`,
:class:`ConceptBackend`) so an Elasticsearch / sqlite implementation can
replace them later without touching harvesting, promotion, or the guided
integration. No external DB dependencies. The default root directory is
``data/graph/``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator, Optional

from patentkit.graph.models import ConceptNode, MatchPair
from patentkit.graph.similarity import normalize_phrase

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "data/graph"


# ---------------------------------------------------------------- interfaces

class MatchPairBackend(ABC):
    """Minimal contract a match-pair store must satisfy."""

    @abstractmethod
    def add(self, pair: MatchPair) -> MatchPair: ...

    @abstractmethod
    def iter(self) -> Iterator[MatchPair]: ...

    @abstractmethod
    def filter(self, **fields) -> list[MatchPair]: ...

    @abstractmethod
    def mark_outcome(self, patent_id: str, outcome: str, *,
                     search_id: Optional[str] = None,
                     feedback_type: Optional[str] = None) -> int: ...


class ConceptBackend(ABC):
    """Minimal contract a concept-graph store must satisfy."""

    @abstractmethod
    def add(self, node: ConceptNode) -> ConceptNode: ...

    @abstractmethod
    def get(self, canonical_name: str) -> Optional[ConceptNode]: ...

    @abstractmethod
    def find_by_alias(self, phrase: str) -> Optional[ConceptNode]: ...

    @abstractmethod
    def nodes(self) -> list[ConceptNode]: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def load(self) -> "ConceptBackend": ...


# -------------------------------------------------------------- match pairs

class MatchPairStore(MatchPairBackend):
    """Append-only JSONL store of :class:`MatchPair` rows.

    ``add`` appends one JSON line per pair (durable immediately);
    ``mark_outcome`` rewrites the file atomically with matching rows
    updated — fine at file-backed scale, and exactly the operation an
    ES/sqlite backend would do with an update query.

    Args:
        root: directory holding ``match_pairs.jsonl`` (default
            ``data/graph/``).
    """

    FILENAME = "match_pairs.jsonl"

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        self.path = self.root / self.FILENAME

    def add(self, pair: MatchPair) -> MatchPair:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(pair.model_dump_json() + "\n")
        return pair

    def add_many(self, pairs: Iterable[MatchPair]) -> int:
        count = 0
        for pair in pairs:
            self.add(pair)
            count += 1
        return count

    def iter(self) -> Iterator[MatchPair]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield MatchPair.model_validate_json(line)
                except ValueError:
                    logger.warning("Skipping malformed match-pair line %d in %s",
                                   line_number, self.path)

    def filter(self, **fields) -> list[MatchPair]:
        """Pairs whose given model fields all equal the given values."""
        return [pair for pair in self.iter()
                if all(getattr(pair, name, None) == value
                       for name, value in fields.items())]

    def mark_outcome(self, patent_id: str, outcome: str, *,
                     search_id: Optional[str] = None,
                     feedback_type: Optional[str] = None) -> int:
        """Set ``outcome`` on every pair matching ``patent_id`` (and
        ``search_id`` when given). ``feedback_type`` is updated only when
        provided. Returns the number of pairs updated."""
        pairs = list(self.iter())
        updated = 0
        for pair in pairs:
            if pair.patent_id != patent_id:
                continue
            if search_id is not None and pair.search_id != search_id:
                continue
            pair.outcome = outcome
            if feedback_type is not None:
                pair.feedback_type = feedback_type
            updated += 1
        if updated:
            self._rewrite(pairs)
        return updated

    def _rewrite(self, pairs: list[MatchPair]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for pair in pairs:
                    fh.write(pair.model_dump_json() + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def __len__(self) -> int:
        return sum(1 for _ in self.iter())


# ------------------------------------------------------------ concept graph

class ConceptGraph(ConceptBackend):
    """In-memory concept graph persisted as one JSON file.

    ``find_by_alias`` matches on the normalized phrase (case/punctuation
    insensitive) against each node's aliases and canonical name.

    Args:
        root: directory holding ``concepts.json`` (default ``data/graph/``).
    """

    FILENAME = "concepts.json"

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        self.path = self.root / self.FILENAME
        self._nodes: dict[str, ConceptNode] = {}

    def add(self, node: ConceptNode) -> ConceptNode:
        self._nodes[node.canonical_name] = node
        return node

    def get(self, canonical_name: str) -> Optional[ConceptNode]:
        return self._nodes.get(canonical_name)

    def find_by_alias(self, phrase: str) -> Optional[ConceptNode]:
        wanted = normalize_phrase(phrase)
        if not wanted:
            return None
        for node in self._nodes.values():
            if wanted == normalize_phrase(node.canonical_name):
                return node
            if any(wanted == normalize_phrase(alias) for alias in node.aliases):
                return node
        return None

    def nodes(self) -> list[ConceptNode]:
        return list(self._nodes.values())

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": [node.model_dump(mode="json")
                             for node in self._nodes.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> "ConceptGraph":
        """Load nodes from disk (no-op if the file does not exist yet)."""
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._nodes = {
                node["canonical_name"]: ConceptNode.model_validate(node)
                for node in payload.get("nodes", [])
            }
        return self

    def __len__(self) -> int:
        return len(self._nodes)


__all__ = ["MatchPairBackend", "MatchPairStore", "ConceptBackend", "ConceptGraph",
           "DEFAULT_ROOT"]
