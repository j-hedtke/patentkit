"""The single tool surface wrapped by both the MCP server and OpenAI tools.

:class:`PatentToolset` methods all return JSON-serializable dicts and their
docstrings double as tool descriptions. :data:`TOOL_SPECS` hand-writes the
JSON-schema for every method (the full :class:`~patentkit.search.base.
SearchQuery` parameter set is enumerated on ``search_patents``), and
:func:`dispatch` routes a (name, arguments) tool call to the right method.

Optional layers (connectors, analysis, viz, evals) are imported lazily
inside methods and degrade into helpful ``{"error": ...}`` dicts, so the
toolset works keys-free with just the in-memory BM25 store.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from patentkit.agents.feedback import SearchFeedback
from patentkit.agents.guided import GuidedSearch, GuidedSearchSession, SessionStore
from patentkit.agents.planner import (
    PlannedQuery,
    QuerySpec,
    SearchPlan,
    estimate_search_seconds,
    humanize_seconds,
)
from patentkit.models import Patent, PatentNumber
from patentkit.search.base import SearchResult
from patentkit.search.bm25 import BM25Store

logger = logging.getLogger(__name__)


def _result_dict(result: SearchResult) -> dict:
    return {
        "patent_number": str(result.patent_number),
        "title": result.patent.title if result.patent else None,
        "score": round(float(result.score), 4),
        "passages": [
            {"text": p.text, "field": p.field, "score": round(float(p.score), 4)}
            for p in result.passages
        ],
        "explanation": result.explanation,
    }


def _session_dict(session: GuidedSearchSession) -> dict:
    return {
        "session_id": session.id,
        "search_type": session.search_type,
        "state": session.state,
        "iteration": session.iteration,
        "plan": session.plan.model_dump(mode="json") if session.plan else None,
        "estimated_seconds": session.params.get("estimated_seconds"),
        "estimated_human": session.params.get("estimated_human"),
        "n_results": len(session.last_results),
    }


class PatentToolset:
    """Every patentkit capability behind one JSON-in / JSON-out surface.

    Args:
        keyword_store: keyword store (default: empty in-memory
            :class:`BM25Store` — index patents via :meth:`index_patents`).
        vector_store: optional vector store for semantic reranking.
        llm: optional LLM; when omitted and ``provider`` is set, a default
            HIGH-effort model for that provider is constructed lazily.
        provider: "anthropic" or "openai" (used only when ``llm`` is None).
        session_dir: directory for guided-session JSON persistence.
        notifiers: notifiers used by :meth:`notify` and search completion.
    """

    def __init__(self, keyword_store=None, vector_store=None, llm=None,
                 provider: Optional[str] = None, session_dir: Optional[str] = None,
                 notifiers: Iterable = ()):
        self.keyword_store = keyword_store if keyword_store is not None else BM25Store()
        self.vector_store = vector_store
        self.provider = provider
        self._llm = llm
        self._llm_resolved = llm is not None
        self.notifiers = list(notifiers)
        self.sessions = SessionStore(session_dir)
        self._guided: Optional[GuidedSearch] = None

    # ------------------------------------------------------------ internals
    @property
    def llm(self):
        """The configured LLM, lazily constructed from ``provider`` — or None
        (keys-free degraded mode) when neither is available."""
        if not self._llm_resolved:
            self._llm_resolved = True
            if self.provider:
                try:
                    from patentkit.llm import get_llm  # noqa: PLC0415
                    self._llm = get_llm("high", provider=self.provider)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not build %s LLM (%s); running keys-free", self.provider, exc)
                    self._llm = None
        return self._llm

    @property
    def guided(self) -> GuidedSearch:
        if self._guided is None:
            self._guided = GuidedSearch(
                keyword_store=self.keyword_store, vector_store=self.vector_store,
                llm=self.llm, session_store=self.sessions,
            )
        return self._guided

    def _fetch_patent(self, number: str) -> Patent:
        """Store first, then the Google Patents connector (lazy import)."""
        found = self.keyword_store.get(PatentNumber.parse(number))
        if found is not None:
            return found
        try:
            from patentkit.connectors.inference.google_patents import fetch_patent  # noqa: PLC0415
        except ImportError as exc:
            raise LookupError(
                f"Patent {number} is not in the local index and the Google Patents "
                "connector is unavailable; index it first with index_patents."
            ) from exc
        return fetch_patent(number)

    def _require_session(self, session_id: str) -> GuidedSearchSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise LookupError(f"No guided session with id {session_id!r}")
        return session

    # ----------------------------------------------------------------- tools
    def search_patents(
        self,
        keywords: Optional[list[str]] = None,
        required_keywords: Optional[list[str]] = None,
        excluded_keywords: Optional[list[str]] = None,
        text: Optional[str] = None,
        minimum_match: Optional[int] = None,
        fields: Optional[list[str]] = None,
        art_classes: Optional[list[str]] = None,
        inventors: Optional[list[str]] = None,
        assignees: Optional[list[str]] = None,
        before_date: Optional[str] = None,
        after_date: Optional[str] = None,
        countries: Optional[list[str]] = None,
        exclude_numbers: Optional[list[str]] = None,
        limit: int = 25,
    ) -> dict:
        """Search the indexed patent corpus with the full query parameter set.

        Keyword search (BM25) over title/abstract/claims/specification with
        metadata filters: required/excluded keywords, minimum-match, CPC art
        class prefixes, inventor/assignee substrings, before/after date
        cutoffs (YYYY-MM-DD), country codes, and an exclude list of patent
        numbers. Returns ranked results with highlighted passages.
        """
        spec = QuerySpec(
            keywords=keywords or [],
            required_keywords=required_keywords or [],
            excluded_keywords=excluded_keywords or [],
            text=text,
            minimum_match=minimum_match,
            fields=fields or ["title", "abstract", "claims", "specification"],
            art_classes=art_classes or [],
            inventors=inventors or [],
            assignees=assignees or [],
            before_date=date.fromisoformat(before_date) if before_date else None,
            after_date=date.fromisoformat(after_date) if after_date else None,
            countries=countries or [],
            exclude_numbers=exclude_numbers or [],
            limit=limit,
        )
        results = self.keyword_store.search(spec.to_search_query())
        return {"count": len(results), "results": [_result_dict(r) for r in results]}

    def get_patent(self, number: str) -> dict:
        """Fetch one patent record (title, abstract, claims, dates, citations,
        classifications) by number, e.g. 'US10123456B2'. Looks in the local
        index first, then falls back to the Google Patents connector."""
        try:
            patent = self._fetch_patent(number)
        except (LookupError, ValueError) as exc:
            return {"error": str(exc)}
        return patent.model_dump(mode="json")

    def index_patents(self, numbers: Optional[list[str]] = None,
                      jsonl_path: Optional[str] = None) -> dict:
        """Add patents to the searchable index, either by fetching the given
        patent numbers (requires the Google Patents connector) or by loading
        canonical Patent JSON records from a .jsonl file (one per line)."""
        patents: list[Patent] = []
        errors: list[str] = []
        if jsonl_path:
            path = Path(jsonl_path)
            if not path.exists():
                return {"error": f"No such file: {jsonl_path}"}
            for i, line in enumerate(path.read_text().splitlines()):
                if not line.strip():
                    continue
                try:
                    patents.append(Patent.model_validate_json(line))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"line {i + 1}: {exc}")
        for number in numbers or []:
            try:
                patents.append(self._fetch_patent(number))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{number}: {exc}")
        indexed = self.keyword_store.index(patents) if patents else 0
        if self.vector_store is not None and patents:
            try:
                self.vector_store.index(patents)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"vector indexing failed: {exc}")
        return {"indexed": indexed, "corpus_size": len(self.keyword_store), "errors": errors}

    def guided_search_start(self, search_type: str, patent_number: Optional[str] = None,
                            product_description: Optional[str] = None,
                            claims: Optional[list[int]] = None) -> dict:
        """Start a guided patent search session. search_type is 'invalidity'
        (prior art against a patent — needs patent_number), 'fto' (freedom to
        operate — needs product_description), or 'infringement' (needs
        patent_number). Returns a session_id, the proposed search plan, and
        an up-front time estimate; present the plan to the user and collect
        feedback before executing."""
        try:
            session = self.guided.start(
                search_type, target_patent_number=patent_number,  # type: ignore[arg-type]
                product_description=product_description, claims=claims,
                fetch=self._fetch_patent if patent_number else None,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return _session_dict(session)

    def guided_search_feedback(self, session_id: str, feedback: dict) -> dict:
        """Apply user feedback to a guided session. The feedback dict may have
        'queries' ([{query_index, verdict: good|too_broad|too_narrow|off_topic,
        note}]), 'results' ([{patent_number, relevant, note}]), 'passages'
        ([{patent_number, passage_text, relevant, note}]), and 'free_text'.
        Plan feedback revises the plan; result feedback queues a refined
        iteration. Returns the updated session state and plan."""
        try:
            session = self._require_session(session_id)
            parsed = SearchFeedback.model_validate(feedback)
            if session.state == "awaiting_result_feedback":
                session = self.guided.apply_result_feedback(session, parsed)
            else:
                session = self.guided.apply_plan_feedback(session, parsed)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return _session_dict(session)

    def guided_search_execute(self, session_id: str) -> dict:
        """Execute the current plan of a guided session (runs the invalidity /
        FTO / infringement agent). Returns ranked results with highlighted
        passages, exclusions applied, and timing. May take the estimated time
        reported by guided_search_start."""
        try:
            session = self._require_session(session_id)
            session = self.guided.execute(session)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        out = _session_dict(session)
        out["results"] = session.last_results
        result = session.params.get("result") or {}
        out["excluded"] = result.get("excluded", {})
        out["timing"] = result.get("timing", {})
        if self.notifiers:
            try:
                from patentkit.notify.base import notify_search_complete  # noqa: PLC0415
                notify_search_complete(self.notifiers, session)
            except Exception:  # noqa: BLE001
                logger.exception("completion notification failed")
        return out

    def guided_search_status(self, session_id: str) -> dict:
        """Get the state of a guided session: its plan, iteration count, time
        estimate, and last results summary."""
        try:
            session = self._require_session(session_id)
        except LookupError as exc:
            return {"error": str(exc)}
        out = _session_dict(session)
        out["results"] = session.last_results
        out["feedback_rounds"] = len(session.feedback_history)
        return out

    def estimate_search_time(self, search_type: str = "invalidity", n_queries: int = 4,
                             corpus_size: Optional[int] = None,
                             charting_claims: int = 0) -> dict:
        """Estimate how long a search will take before running it, from the
        number of planned queries, the corpus size (defaults to the local
        index size), whether LLM reranking is on, and how many claims will be
        charted. Returns seconds and a human-readable duration."""
        if corpus_size is None:
            corpus_size = max(len(self.keyword_store), 1)
        if search_type not in ("invalidity", "fto", "infringement"):
            search_type = "invalidity"
        plan = SearchPlan(
            search_type=search_type,  # type: ignore[arg-type]
            target="estimate", rationale="estimate",
            queries=[PlannedQuery(purpose="q", query=QuerySpec()) for _ in range(max(1, n_queries))],
        )
        seconds = estimate_search_seconds(plan, corpus_size=corpus_size,
                                          with_llm_rerank=self.llm is not None,
                                          charting_claims=charting_claims)
        return {"seconds": seconds, "human": humanize_seconds(seconds),
                "corpus_size": corpus_size, "llm_rerank": self.llm is not None}

    def build_claim_chart(self, patent_number: str, claim_number: int,
                          reference_numbers: list[str]) -> dict:
        """Build an element-by-element invalidity claim chart mapping one claim
        of the target patent against the given prior-art references. Requires
        the patentkit analysis module; references must be fetchable."""
        try:
            from patentkit.analysis.invalidity import build_claim_chart  # noqa: PLC0415
        except ImportError:
            return {"error": "Claim charting requires patentkit.analysis "
                             "(not available in this installation)."}
        try:
            patent = self._fetch_patent(patent_number)
            references = [self._fetch_patent(n) for n in reference_numbers]
            chart = build_claim_chart(patent, claim_number, references, self.llm)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        data: dict[str, Any]
        if hasattr(chart, "model_dump"):
            data = chart.model_dump(mode="json")
        else:
            data = {k: str(v) for k, v in vars(chart).items() if not k.startswith("_")}
        try:
            data["coverage_summary"] = chart.coverage_summary()
        except Exception as exc:  # noqa: BLE001
            data["coverage_summary"] = f"unavailable: {exc}"
        return data

    def cluster_patents(self, numbers: Optional[list[str]] = None,
                        query: Optional[str] = None) -> dict:
        """Cluster a set of patents (by number list, or the results of a
        keyword query) into technology topics. Requires the viz extra
        (pip install 'patentkit[viz]')."""
        try:
            from patentkit.viz.clustering import cluster_patents as _cluster  # noqa: PLC0415
        except ImportError:
            return {"error": "Clustering requires the viz extra: pip install 'patentkit[viz]'"}
        try:
            patents: list[Patent] = []
            if numbers:
                patents = [self._fetch_patent(n) for n in numbers]
            elif query:
                hits = self.search_patents(keywords=query.split(), limit=100)["results"]
                patents = [self._fetch_patent(h["patent_number"]) for h in hits]
            from patentkit.search.vector import HashingEmbeddings  # noqa: PLC0415
            result = _cluster(patents, HashingEmbeddings(), llm=self.llm)
            return {
                "labels": list(result.labels),
                "topics": {str(k): v for k, v in result.topics.items()},
                "silhouette": result.silhouette,
                "representative": {str(k): v for k, v in result.representative.items()},
            }
        except ImportError as exc:  # numpy/sklearn missing
            return {"error": f"Clustering requires the viz extra (pip install 'patentkit[viz]'): {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def run_eval(self, dataset_path: Optional[str] = None) -> dict:
        """Run the search-performance eval harness over a dataset (default:
        the built-in toy dataset). Requires the patentkit evals module."""
        try:
            from patentkit.evals.datasets import (  # noqa: PLC0415
                default_ipr_toy_dataset,
                load_queryset_jsonl,
            )
            from patentkit.evals.harness import EvalRunner, searchfn_from_stores  # noqa: PLC0415
        except ImportError:
            return {"error": "Evals require the patentkit.evals module "
                             "(not available in this installation)."}
        try:
            dataset = load_queryset_jsonl(dataset_path) if dataset_path else default_ipr_toy_dataset()
            runner = EvalRunner(searchfn_from_stores(self.keyword_store), dataset,
                                name="patentkit-eval")
            report = runner.run()
            return {
                "name": report.name,
                "aggregates": report.aggregates,
                "mean_recall_curve": report.mean_curve,
                "queries": len(report.rows),
                "errors": report.errors,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def notify(self, subject: str, body: str) -> dict:
        """Send a notification (Slack/email, as configured on this server)
        with the given subject and body — e.g. to announce that a long search
        finished and where to find the results."""
        if not self.notifiers:
            return {"sent": 0, "note": "no notifiers configured"}
        sent, errors = 0, []
        for notifier in self.notifiers:
            try:
                notifier.send(subject, body)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("notifier failed")
                errors.append(f"{type(notifier).__name__}: {exc}")
        return {"sent": sent, "errors": errors}


# --------------------------------------------------------------- tool specs

def _arr(description: str, item_type: str = "string") -> dict:
    return {"type": "array", "items": {"type": item_type}, "description": description}


_SEARCH_QUERY_PROPERTIES: dict[str, dict] = {
    "keywords": _arr("Search keywords/phrases (OR semantics with minimum_match)."),
    "required_keywords": _arr("Keywords that MUST all appear (AND semantics)."),
    "excluded_keywords": _arr("Tokens/phrases that must NOT appear anywhere."),
    "text": {"type": "string", "description": "Free-text query for phrase/semantic matching."},
    "minimum_match": {"type": "integer",
                      "description": "Minimum number of `keywords` that must match (default len//3, >=1)."},
    "fields": _arr("Fields to search; subset of title, abstract, claims, specification."),
    "art_classes": _arr("CPC/IPC art-class prefixes, e.g. ['G06F16', 'H04L']."),
    "inventors": _arr("Inventor-name substrings to require."),
    "assignees": _arr("Assignee-name substrings to require."),
    "before_date": {"type": "string", "description": "Only documents effective before this date "
                                                     "(YYYY-MM-DD) — the prior-art cutoff."},
    "after_date": {"type": "string", "description": "Only documents effective after this date (YYYY-MM-DD)."},
    "countries": _arr("Country codes to allow, e.g. ['US', 'EP']."),
    "exclude_numbers": _arr("Patent numbers to exclude from results."),
    "limit": {"type": "integer", "description": "Maximum results to return.", "default": 25},
}

_FEEDBACK_SCHEMA: dict = {
    "type": "object",
    "description": "Structured user feedback.",
    "properties": {
        "queries": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "query_index": {"type": "integer"},
                "verdict": {"type": "string",
                            "enum": ["good", "too_broad", "too_narrow", "off_topic"]},
                "note": {"type": "string"},
            },
            "required": ["query_index", "verdict"],
        }},
        "results": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "patent_number": {"type": "string"},
                "relevant": {"type": "boolean"},
                "note": {"type": "string"},
            },
            "required": ["patent_number"],
        }},
        "passages": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "patent_number": {"type": "string"},
                "passage_text": {"type": "string"},
                "relevant": {"type": "boolean"},
                "note": {"type": "string"},
            },
            "required": ["patent_number", "passage_text"],
        }},
        "free_text": {"type": "string"},
    },
}


def _spec(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


#: One entry per PatentToolset method; the single source of truth for both
#: the MCP server and the OpenAI function-calling layer.
TOOL_SPECS: list[dict] = [
    _spec(
        "search_patents",
        "Search the indexed patent corpus (BM25 keyword search with metadata "
        "filters). Returns ranked patents with highlighted passages. Supports "
        "the full query parameter set: keywords, required/excluded keywords, "
        "free text, minimum-match, field selection, CPC art classes, "
        "inventors, assignees, date cutoffs, countries, and exclusions.",
        dict(_SEARCH_QUERY_PROPERTIES),
    ),
    _spec(
        "get_patent",
        "Fetch one patent record (title, abstract, claims, dates, citations, "
        "classifications) by number, e.g. 'US10123456B2'. Checks the local "
        "index first, then Google Patents when that connector is available.",
        {"number": {"type": "string", "description": "Patent number, e.g. 'US10123456B2'."}},
        ["number"],
    ),
    _spec(
        "index_patents",
        "Add patents to the searchable index, by patent numbers (fetched via "
        "the Google Patents connector) and/or a JSONL file of canonical "
        "Patent records (one JSON object per line).",
        {
            "numbers": _arr("Patent numbers to fetch and index."),
            "jsonl_path": {"type": "string", "description": "Path to a .jsonl file of Patent records."},
        },
    ),
    _spec(
        "guided_search_start",
        "Start a guided patent search session ('invalidity' needs "
        "patent_number; 'fto' needs product_description; 'infringement' needs "
        "patent_number). Returns a session_id, a proposed multi-query search "
        "plan, and an up-front time estimate. Present the plan (and estimate) "
        "to the user and collect feedback before executing. Invalidity "
        "searches exclude examiner-cited art and family members by default.",
        {
            "search_type": {"type": "string", "enum": ["invalidity", "fto", "infringement"],
                            "description": "Which search workflow to run."},
            "patent_number": {"type": "string", "description": "Target patent number."},
            "product_description": {"type": "string", "description": "Target product (FTO)."},
            "claims": _arr("Claim numbers in focus (default: independent claims).", "integer"),
        },
        ["search_type"],
    ),
    _spec(
        "guided_search_feedback",
        "Apply user feedback to a guided session. Before execution it revises "
        "the plan; after execution it queues a refined iteration (irrelevant "
        "results become exclusions, query verdicts adjust breadth). Returns "
        "the updated state and plan.",
        {
            "session_id": {"type": "string", "description": "Session id from guided_search_start."},
            "feedback": _FEEDBACK_SCHEMA,
        },
        ["session_id", "feedback"],
    ),
    _spec(
        "guided_search_execute",
        "Execute the current plan of a guided session: runs the 3-stage "
        "invalidity pipeline (keyword -> semantic rerank -> LLM scoring), the "
        "FTO screen, or the infringement candidate ranking. Returns ranked "
        "results with highlighted passages, what was excluded and why, and "
        "per-stage timing.",
        {"session_id": {"type": "string", "description": "Session id from guided_search_start."}},
        ["session_id"],
    ),
    _spec(
        "guided_search_status",
        "Get the state of a guided session: plan, iteration, time estimate, "
        "feedback rounds, and the latest ranked results.",
        {"session_id": {"type": "string", "description": "Session id from guided_search_start."}},
        ["session_id"],
    ),
    _spec(
        "estimate_search_time",
        "Estimate how long a search will take before running it (per-query "
        "base + corpus-size factor + LLM rerank latency + per-claim charting "
        "cost). Returns seconds and a human-readable duration.",
        {
            "search_type": {"type": "string", "enum": ["invalidity", "fto", "infringement"],
                            "default": "invalidity"},
            "n_queries": {"type": "integer", "description": "Planned query count.", "default": 4},
            "corpus_size": {"type": "integer",
                            "description": "Corpus size (default: local index size)."},
            "charting_claims": {"type": "integer",
                                "description": "Claims that will be charted afterwards.", "default": 0},
        },
    ),
    _spec(
        "build_claim_chart",
        "Build an element-by-element invalidity claim chart mapping one claim "
        "of the target patent against prior-art references, with a coverage "
        "summary of which limitations each reference discloses.",
        {
            "patent_number": {"type": "string", "description": "Target patent number."},
            "claim_number": {"type": "integer", "description": "Claim to chart."},
            "reference_numbers": _arr("Prior-art reference patent numbers."),
        },
        ["patent_number", "claim_number", "reference_numbers"],
    ),
    _spec(
        "cluster_patents",
        "Cluster a set of patents into technology topics (requires the viz "
        "extra). Provide explicit patent numbers or a keyword query whose "
        "results will be clustered.",
        {
            "numbers": _arr("Patent numbers to cluster."),
            "query": {"type": "string", "description": "Keyword query selecting patents to cluster."},
        },
    ),
    _spec(
        "run_eval",
        "Run the search-performance eval harness (precision/recall against "
        "labeled prior art). Uses the built-in toy dataset unless a dataset "
        "path is given.",
        {"dataset_path": {"type": "string", "description": "Path to an eval dataset file."}},
    ),
    _spec(
        "notify",
        "Send a notification through the configured channels (Slack webhook, "
        "SendGrid/SMTP email) — e.g. to announce a finished long-running "
        "search.",
        {
            "subject": {"type": "string", "description": "Short subject/headline."},
            "body": {"type": "string", "description": "Message body."},
        },
        ["subject", "body"],
    ),
]

_TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


def dispatch(toolset: PatentToolset, name: str, arguments: dict) -> dict:
    """Route one tool call to the toolset; always returns a JSON-able dict.

    Unknown tools and unexpected method errors come back as
    ``{"error": ...}`` so transport layers never crash on a bad call.
    """
    if name not in _TOOL_NAMES:
        return {"error": f"Unknown tool {name!r}. Known tools: {sorted(_TOOL_NAMES)}"}
    method = getattr(toolset, name)
    try:
        result = method(**(arguments or {}))
    except TypeError as exc:  # bad/missing arguments
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 — transport layers must not crash
        logger.exception("tool %s failed", name)
        return {"error": f"{name} failed: {exc}"}
    # Defensive: guarantee JSON serializability for transport layers.
    return json.loads(json.dumps(result, default=str))


__all__ = ["PatentToolset", "TOOL_SPECS", "dispatch"]
