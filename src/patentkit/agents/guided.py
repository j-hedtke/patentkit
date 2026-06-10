"""The user-facing guided search loop.

A :class:`GuidedSearchSession` is a fully serializable state machine
(``model_dump_json`` / ``model_validate_json``) so MCP and OpenAI tool
layers can drive it across conversation turns:

    planning -> awaiting_plan_feedback -> searching ->
    awaiting_result_feedback -> (searching ...) -> done

Execution runs the agentic search core: one LLM agent conversation issues
and refines its own queries under step/wall-clock budgets. The session
persists the agent's reasoning trace and the resumable conversation, so
result-stage feedback is NOT a plan rewrite — it is queued and injected as a
user message when execution resumes the SAME agent conversation. Plan-stage
feedback (before the first execution) heuristically adjusts the pre-run
preview and seeds the agent's initial guidance.

Typical flow::

    guided = GuidedSearch(keyword_store=store, llm=get_llm("high"))
    session = guided.start("invalidity", target_patent_number="US10123456B2")
    # ... show session.plan + estimated time, collect SearchFeedback ...
    session = guided.apply_plan_feedback(session, feedback)
    session = guided.execute(session)          # runs the agent
    # ... show session.last_results + trace, collect feedback ...
    session = guided.apply_result_feedback(session, feedback)
    session = guided.execute(session)          # resumes the SAME conversation
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

from patentkit.agents.agentic import SearchTrace
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

logger = logging.getLogger(__name__)

SessionState = Literal[
    "planning", "awaiting_plan_feedback", "searching", "awaiting_result_feedback", "done"
]


class GuidedSearchSession(BaseModel):
    """Serializable state of one guided search.

    ``params`` carries everything execution needs — the target patent as a
    JSON dump, claim selection, product description, time estimates, the
    last full agent result, the agent's reasoning trace
    (``params["trace"]``), the resumable agent conversation
    (``params["conversation"]``), and queued feedback messages
    (``params["pending_feedback"]``) — so a session restored from JSON in a
    later process is fully executable and resumable.
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
    """Drives guided sessions: plan preview -> feedback -> agentic execution.

    Args:
        keyword_store: keyword store searched by the agents.
        vector_store: optional vector store (adds the agent's
            ``semantic_search`` tool).
        llm: optional LLM driving the agent; ``None`` runs everything in
            keys-free degraded mode (single keyword pass per execution).
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
            search_type: which agent will execute the search.
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
        """Like :meth:`start` but with an already-resolved :class:`Patent`.

        The plan preview is derived deterministically (no LLM call), so
        starting a session is instant.
        """
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
                "pending_feedback": [],
                "pre_run_guidance": [],
            },
        )
        return self._save(session)

    # -------------------------------------------------------------- feedback
    def apply_plan_feedback(self, session: GuidedSearchSession,
                            feedback: SearchFeedback) -> GuidedSearchSession:
        """Apply pre-run feedback: heuristically adjust the plan preview and
        seed the agent's initial guidance. Stays in awaiting_plan_feedback so
        the user can review (or call :meth:`execute` directly)."""
        if session.plan is None:
            raise ValueError(f"Session {session.id} has no plan to revise")
        session.plan = revise_plan(session.plan, feedback)
        session.feedback_history.append(feedback)
        guidance = session.params.setdefault("pre_run_guidance", [])
        guidance.append(feedback.summary_for_llm())
        session.state = "awaiting_plan_feedback"
        return self._save(session)

    def apply_result_feedback(self, session: GuidedSearchSession,
                              feedback: SearchFeedback) -> GuidedSearchSession:
        """Queue result feedback for injection into the resumed agent
        conversation, and queue another iteration.

        No plan rewrite / LLM call happens here: the feedback summary is
        injected as a user message when :meth:`execute` resumes the SAME
        conversation. Results marked irrelevant additionally become hard
        exclusions enforced at the tool layer (and in degraded mode).
        """
        if session.plan is None:
            raise ValueError(f"Session {session.id} has no plan; call start() first")
        session.feedback_history.append(feedback)
        pending = session.params.setdefault("pending_feedback", [])
        pending.append(feedback.summary_for_llm())
        for result in feedback.results:  # hard exclusions, enforced at the tool layer
            if result.relevant is False and result.patent_number not in session.plan.exclusions:
                session.plan.exclusions.append(result.patent_number)
        session.iteration += 1
        session.state = "searching"
        return self._save(session)

    # --------------------------------------------------------------- execute
    def execute(self, session: GuidedSearchSession,
                progress: Optional[Callable[[str], None]] = None) -> GuidedSearchSession:
        """Run (or resume) the agentic search for this session.

        First execution starts a fresh agent conversation seeded with the
        plan's starting angles and any pre-run guidance; later executions
        resume the persisted conversation with the queued feedback messages
        injected. The reasoning trace lands in ``session.params["trace"]``,
        the resumable conversation in ``session.params["conversation"]``,
        ranked results in ``session.last_results``, and the full agent
        result model in ``session.params["result"]``.
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

        patent = self._session_patent(session)
        feedback_messages = self._drain_feedback(session)
        resume = session.params.get("conversation") or None

        if session.search_type == "invalidity":
            if patent is None:
                raise ValueError("Invalidity search needs a target patent")
            agent = InvaliditySearchAgent(self.keyword_store, self.vector_store, self.llm)
            result = agent.search(
                patent, claims=session.params.get("claims"),
                custom_exclusions=session.plan.exclusions,
                feedback_messages=feedback_messages, resume_messages=resume,
                progress=progress,
            )
        elif session.search_type == "fto":
            description = session.params.get("product_description")
            if not description:
                raise ValueError("FTO search needs a product_description")
            agent = FtoSearchAgent(self.keyword_store, self.vector_store, self.llm)
            result = agent.search(
                description, custom_exclusions=session.plan.exclusions,
                feedback_messages=feedback_messages, resume_messages=resume,
                progress=progress,
            )
        else:  # infringement
            if patent is None:
                raise ValueError("Infringement search needs a target patent")
            agent = InfringementSearchAgent(self.llm, keyword_store=self.keyword_store)
            result = agent.search(
                patent, claims=session.params.get("claims"),
                product_candidates=session.params.get("product_candidates"),
                feedback_messages=feedback_messages, resume_messages=resume,
                progress=progress,
            )

        session.last_results = result.results
        dump = result.model_dump(mode="json")
        session.params["conversation"] = dump.pop("conversation", None)
        session.params["trace"] = dump.get("trace")
        session.params["result"] = dump
        session.params["stop_reason"] = result.stop_reason
        session.params["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        session.state = "awaiting_result_feedback"
        return self._save(session)

    def finish(self, session: GuidedSearchSession) -> GuidedSearchSession:
        """Mark the session done."""
        session.state = "done"
        return self._save(session)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _drain_feedback(session: GuidedSearchSession) -> list[str]:
        """Consume queued pre-run guidance + result feedback for injection."""
        messages: list[str] = []
        if not session.params.get("conversation"):
            # first run: seed the agent with the plan's angles + plan feedback
            plan = session.plan
            if plan is not None and plan.queries:
                angles = "; ".join(
                    f"{q.purpose}: {', '.join(q.query.keywords[:8])}" for q in plan.queries)
                messages.append("Suggested starting angles (from the reviewed plan): " + angles)
            messages += session.params.get("pre_run_guidance") or []
            session.params["pre_run_guidance"] = []
        messages += session.params.get("pending_feedback") or []
        session.params["pending_feedback"] = []
        return messages

    @staticmethod
    def _session_patent(session: GuidedSearchSession) -> Optional[Patent]:
        dump = session.params.get("patent")
        return Patent.model_validate(dump) if dump else None

    @staticmethod
    def session_trace(session: GuidedSearchSession) -> Optional[SearchTrace]:
        """The persisted reasoning trace of the last execution, if any."""
        dump = session.params.get("trace")
        return SearchTrace.model_validate(dump) if dump else None


def restore_session(payload: str | dict) -> GuidedSearchSession:
    """Restore a session from ``model_dump_json()`` output (str or dict)."""
    if isinstance(payload, str):
        return GuidedSearchSession.model_validate_json(payload)
    return GuidedSearchSession.model_validate(json.loads(json.dumps(payload)))


__all__ = ["GuidedSearch", "GuidedSearchSession", "SessionStore", "restore_session"]
