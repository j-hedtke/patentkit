"""The user-facing guided search loop.

A :class:`GuidedSearchSession` is a fully serializable state machine
(``model_dump_json`` / ``model_validate_json``) so MCP and OpenAI tool
layers can drive it across conversation turns:

    planning -> awaiting_plan_feedback -> searching ->
    awaiting_result_feedback -> (searching ...) -> done

Typical flow::

    guided = GuidedSearch(keyword_store=store, llm=get_llm("high"))
    session = guided.start("invalidity", target_patent_number="US10123456B2")
    # ... show session.plan + estimated time, collect SearchFeedback ...
    session = guided.apply_plan_feedback(session, feedback)
    session = guided.execute(session)          # runs the right agent
    # ... show session.last_results, collect feedback ...
    session = guided.apply_result_feedback(session, feedback)
    session = guided.execute(session)          # next iteration
    session = guided.finish(session)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from patentkit.agents.feedback import SearchFeedback
from patentkit.agents.fto_search import FtoSearchAgent
from patentkit.agents.infringement_search import InfringementSearchAgent
from patentkit.agents.invalidity_search import InvaliditySearchAgent
from patentkit.agents.planner import (
    SearchPlan,
    estimate_search_seconds,
    humanize_seconds,
    plan_search,
    revise_plan,
)
from patentkit.models import Patent, PatentNumber
from patentkit.search.base import SearchQuery

logger = logging.getLogger(__name__)

SessionState = Literal[
    "planning", "awaiting_plan_feedback", "searching", "awaiting_result_feedback", "done"
]


class GuidedSearchSession(BaseModel):
    """Serializable state of one guided search.

    ``params`` carries everything execution needs (the target patent as a
    JSON dump, claim selection, product description, time estimates, and the
    last full agent result), so a session restored from JSON in a later
    process is fully executable.
    """

    id: str
    search_type: Literal["invalidity", "fto", "infringement"]
    state: SessionState = "planning"
    plan: Optional[SearchPlan] = None
    last_results: list[dict] = Field(default_factory=list)
    feedback_history: list[SearchFeedback] = Field(default_factory=list)
    iteration: int = 0
    params: dict = Field(default_factory=dict)


class SessionStore:
    """In-memory session registry with optional JSON directory persistence."""

    def __init__(self, directory: str | Path | None = None):
        self._sessions: dict[str, GuidedSearchSession] = {}
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, session: GuidedSearchSession) -> None:
        self._sessions[session.id] = session
        if self.directory:
            (self.directory / f"{session.id}.json").write_text(session.model_dump_json())

    def get(self, session_id: str) -> Optional[GuidedSearchSession]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        if self.directory:
            path = self.directory / f"{session_id}.json"
            if path.exists():
                session = GuidedSearchSession.model_validate_json(path.read_text())
                self._sessions[session_id] = session
                return session
        return None

    def list_ids(self) -> list[str]:
        ids = set(self._sessions)
        if self.directory:
            ids |= {p.stem for p in self.directory.glob("*.json")}
        return sorted(ids)


class GuidedSearch:
    """Drives guided sessions: plan -> feedback -> execute -> iterate.

    Args:
        keyword_store: keyword store searched by the agents.
        vector_store: optional vector store for semantic reranking.
        llm: optional LLM used for planning, revision, and stage-3 scoring;
            ``None`` runs everything in keys-free degraded mode.
        session_store: optional :class:`SessionStore`; sessions are saved
            into it after every transition when provided.
    """

    def __init__(self, keyword_store=None, vector_store=None, llm=None,
                 session_store: Optional[SessionStore] = None):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.llm = llm
        self.session_store = session_store

    # ------------------------------------------------------------- plumbing
    def _save(self, session: GuidedSearchSession) -> GuidedSearchSession:
        if self.session_store is not None:
            self.session_store.save(session)
        return session

    def _resolve_patent(self, number: str, fetch: Optional[Callable[[str], Patent]]) -> Patent:
        """Resolve a patent: explicit fetch callable, then the keyword store,
        then the Google Patents connector (lazy import)."""
        if fetch is not None:
            return fetch(number)
        if self.keyword_store is not None:
            found = self.keyword_store.get(PatentNumber.parse(number))
            if found is not None:
                return found
        try:
            from patentkit.connectors.inference.google_patents import fetch_patent  # noqa: PLC0415
        except ImportError as exc:
            raise ValueError(
                f"Patent {number} is not in the local store and the Google Patents "
                "connector is unavailable. Index it first, pass fetch=..., or use "
                "start_with_patent(...)."
            ) from exc
        return fetch_patent(number)

    # ----------------------------------------------------------------- start
    def start(
        self,
        search_type: Literal["invalidity", "fto", "infringement"],
        target_patent_number: Optional[str] = None,
        product_description: Optional[str] = None,
        claims: Optional[list[int]] = None,
        fetch: Optional[Callable[[str], Patent]] = None,
    ) -> GuidedSearchSession:
        """Create a session and plan it; state becomes awaiting_plan_feedback.

        Args:
            search_type: which agent will execute the plan.
            target_patent_number: target patent (invalidity / infringement);
                resolved via ``fetch``, then the keyword store, then the
                Google Patents connector.
            product_description: target product (fto / infringement context).
            claims: claim numbers in focus.
            fetch: optional ``number -> Patent`` resolver override.
        """
        patent: Optional[Patent] = None
        if target_patent_number:
            patent = self._resolve_patent(target_patent_number, fetch)
        return self.start_with_patent(search_type, patent,
                                      product_description=product_description, claims=claims)

    def start_with_patent(
        self,
        search_type: Literal["invalidity", "fto", "infringement"],
        patent: Optional[Patent],
        *,
        product_description: Optional[str] = None,
        claims: Optional[list[int]] = None,
    ) -> GuidedSearchSession:
        """Like :meth:`start` but with an already-resolved :class:`Patent`."""
        plan = plan_search(search_type, patent=patent,
                           product_description=product_description, claims=claims, llm=self.llm)
        corpus = len(self.keyword_store) if self.keyword_store is not None else 0
        estimated = estimate_search_seconds(plan, corpus_size=corpus,
                                            with_llm_rerank=self.llm is not None)
        plan.estimated_seconds = estimated
        session = GuidedSearchSession(
            id=uuid.uuid4().hex[:12],
            search_type=search_type,
            state="awaiting_plan_feedback",
            plan=plan,
            params={
                "patent": patent.model_dump(mode="json") if patent else None,
                "claims": claims,
                "product_description": product_description,
                "estimated_seconds": estimated,
                "estimated_human": humanize_seconds(estimated),
            },
        )
        return self._save(session)

    # -------------------------------------------------------------- feedback
    def apply_plan_feedback(self, session: GuidedSearchSession,
                            feedback: SearchFeedback) -> GuidedSearchSession:
        """Revise the plan from feedback; stays in awaiting_plan_feedback so
        the user can review the revision (or call :meth:`execute` directly)."""
        if session.plan is None:
            raise ValueError(f"Session {session.id} has no plan to revise")
        session.plan = revise_plan(session.plan, feedback, llm=self.llm)
        session.feedback_history.append(feedback)
        session.state = "awaiting_plan_feedback"
        return self._save(session)

    def apply_result_feedback(self, session: GuidedSearchSession,
                              feedback: SearchFeedback) -> GuidedSearchSession:
        """Revise the plan from result feedback and queue another iteration."""
        if session.plan is None:
            raise ValueError(f"Session {session.id} has no plan to revise")
        session.plan = revise_plan(session.plan, feedback, llm=self.llm)
        session.feedback_history.append(feedback)
        session.iteration += 1
        session.state = "searching"
        return self._save(session)

    # --------------------------------------------------------------- execute
    def execute(self, session: GuidedSearchSession,
                progress: Optional[Callable[[str], None]] = None) -> GuidedSearchSession:
        """Run the right agent for the session's plan.

        The plan's queries are folded into one extra :class:`SearchQuery`
        (keyword/required/excluded/art-class union, earliest before_date) so
        complementary angles widen stage-1 recall; the plan's exclusions are
        applied as custom exclusions. Results land in ``session.last_results``
        and the full agent result model in ``session.params["result"]``.
        """
        if session.plan is None:
            raise ValueError(f"Session {session.id} has no plan; call start() first")
        if session.state == "done":
            raise ValueError(f"Session {session.id} is already done")
        session.state = "searching"
        t0 = time.monotonic()

        corpus = len(self.keyword_store) if self.keyword_store is not None else 0
        session.params["estimated_seconds"] = estimate_search_seconds(
            session.plan, corpus_size=corpus, with_llm_rerank=self.llm is not None)
        session.params["estimated_human"] = humanize_seconds(session.params["estimated_seconds"])

        extra = self._fold_plan_queries(session.plan)
        patent = self._session_patent(session)

        if session.search_type == "invalidity":
            if patent is None:
                raise ValueError("Invalidity search needs a target patent")
            agent = InvaliditySearchAgent(self.keyword_store, self.vector_store, self.llm)
            result = agent.search(
                patent, claims=session.params.get("claims"), extra_query=extra,
                custom_exclusions=session.plan.exclusions, progress=progress,
            )
        elif session.search_type == "fto":
            description = session.params.get("product_description")
            if not description:
                raise ValueError("FTO search needs a product_description")
            agent = FtoSearchAgent(self.keyword_store, self.vector_store, self.llm)
            result = agent.search(description, extra_query=extra, progress=progress)
        else:  # infringement
            if patent is None:
                raise ValueError("Infringement search needs a target patent")
            agent = InfringementSearchAgent(self.llm)
            result = agent.search(
                patent, claims=session.params.get("claims"),
                product_candidates=session.params.get("product_candidates"),
                progress=progress,
            )

        session.last_results = result.results
        session.params["result"] = result.model_dump(mode="json")
        session.params["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        session.state = "awaiting_result_feedback"
        return self._save(session)

    def finish(self, session: GuidedSearchSession) -> GuidedSearchSession:
        """Mark the session done."""
        session.state = "done"
        return self._save(session)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _session_patent(session: GuidedSearchSession) -> Optional[Patent]:
        dump = session.params.get("patent")
        return Patent.model_validate(dump) if dump else None

    @staticmethod
    def _fold_plan_queries(plan: SearchPlan) -> Optional[SearchQuery]:
        """Union the plan's queries into one extra SearchQuery for the agents."""
        if not plan.queries:
            return None
        folded = plan.queries[0].query.to_search_query()
        for planned in plan.queries[1:]:
            from patentkit.agents._support import merge_query  # local helper
            folded = merge_query(folded, planned.query.to_search_query())
        return folded


def restore_session(payload: str | dict) -> GuidedSearchSession:
    """Restore a session from ``model_dump_json()`` output (str or dict)."""
    if isinstance(payload, str):
        return GuidedSearchSession.model_validate_json(payload)
    return GuidedSearchSession.model_validate(json.loads(json.dumps(payload)))


__all__ = ["GuidedSearch", "GuidedSearchSession", "SessionStore", "restore_session"]
