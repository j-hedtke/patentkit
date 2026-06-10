"""Pure agentic search core.

An LLM agent on the provider's native tool-use platform (Anthropic Messages
API tool use; OpenAI Responses API function tools) drives the search itself:
it generates queries, executes them as tools against the configured stores,
reads the results, refines its angles, maintains a working shortlist, and
decides when to stop — finishing with a ranked candidate list. Runs complete
in seconds to minutes under explicit step and wall-clock budgets, produce a
full saved reasoning trace (:class:`SearchTrace`), and expose the resumable
conversation so user feedback can be injected and the SAME agent
conversation continued.

Hard guarantees enforced in the tool layer (not just the prompt):

- the search-type exclusion list (examiner-cited art, family, the target
  itself, custom) is ALWAYS applied — excluded numbers are never returned by
  the search tools and are rejected from shortlist/finish;
- for invalidity, ``before_date`` is clamped to the prior-art cutoff.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, Field

from patentkit.llm.tools import (
    ToolDef,
    TraceStep,
    run_tool_loop,
    user_text_message,
)
from patentkit.models import Claim, Patent, PatentNumber
from patentkit.search.base import KeywordStore, SearchQuery, SearchResult, VectorStore

logger = logging.getLogger(__name__)

#: defaults sized so a search finishes well inside 3 minutes
DEFAULT_MAX_STEPS = 16
DEFAULT_BUDGET_SECONDS = 180.0

#: cap on per-search-tool-call results fed back to the model
MAX_TOOL_RESULTS = 25
#: passage snippet length in compact tool results
SNIPPET_CHARS = 220


# --------------------------------------------------------------------- trace

class TraceStepModel(BaseModel):
    """Pydantic mirror of :class:`patentkit.llm.tools.TraceStep`."""

    index: int
    kind: str
    content: str
    tool_name: Optional[str] = None
    arguments: Optional[dict] = None
    elapsed_s: float = 0.0

    @classmethod
    def from_step(cls, step: TraceStep) -> "TraceStepModel":
        return cls(**step.to_dict())


class SearchTrace(BaseModel):
    """The full reasoning trace of one agentic search run.

    Wraps the loop's trace steps plus the budgets, the queries the agent
    issued, every shortlist revision, injected user-feedback events, and
    timings. ``save()`` writes JSON; ``to_markdown()`` renders a
    human-readable reasoning trace.
    """

    search_type: str
    target: Optional[str] = None
    max_steps: int = DEFAULT_MAX_STEPS
    budget_seconds: float = DEFAULT_BUDGET_SECONDS
    stop_reason: str = ""
    elapsed_s: float = 0.0
    steps: list[TraceStepModel] = Field(default_factory=list)
    #: user-feedback messages injected into the conversation, in order
    feedback: list[str] = Field(default_factory=list)
    #: every search query the agent issued: {"tool": ..., "arguments": {...}}
    queries: list[dict] = Field(default_factory=list)
    #: every shortlist revision the agent recorded, in order
    shortlist_history: list[list[dict]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Write the trace as JSON; returns the path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2))
        return out

    def to_markdown(self) -> str:
        """Render a human-readable reasoning trace."""
        lines = [
            f"# Agentic {self.search_type} search trace",
            "",
            f"- target: {self.target or '(none)'}",
            f"- budgets: max_steps={self.max_steps}, budget_seconds={self.budget_seconds:g}",
            f"- stop_reason: {self.stop_reason or '(running)'}; elapsed {self.elapsed_s:.1f}s",
            f"- queries issued: {len(self.queries)}; shortlist revisions: "
            f"{len(self.shortlist_history)}",
            "",
        ]
        if self.feedback:
            lines.append("## Injected user feedback")
            lines += [f"- {f}" for f in self.feedback]
            lines.append("")
        lines.append("## Steps")
        for step in self.steps:
            label = step.kind + (f" — {step.tool_name}" if step.tool_name else "")
            lines.append(f"### Step {step.index} [{step.elapsed_s:.1f}s] {label}")
            if step.kind == "tool_call" and step.arguments is not None:
                lines.append("```json")
                lines.append(json.dumps(step.arguments, indent=2, default=str))
                lines.append("```")
            else:
                content = step.content
                if len(content) > 1500:
                    content = content[:1500] + " … [truncated]"
                lines.append(content)
            lines.append("")
        if self.queries:
            lines.append("## Queries issued")
            for i, query in enumerate(self.queries, start=1):
                lines.append(f"{i}. `{query['tool']}` "
                             f"{json.dumps(query['arguments'], default=str)}")
            lines.append("")
        if self.shortlist_history:
            lines.append("## Shortlist evolution")
            for i, snapshot in enumerate(self.shortlist_history, start=1):
                numbers = ", ".join(str(c.get("number", "?")) for c in snapshot)
                lines.append(f"- revision {i}: {numbers}")
            lines.append("")
        return "\n".join(lines)

    def summary(self) -> dict:
        """Compact JSON summary for status displays."""
        return {
            "step_count": len(self.steps),
            "queries_issued": [q["arguments"] for q in self.queries],
            "shortlist": self.shortlist_history[-1] if self.shortlist_history else [],
            "stop_reason": self.stop_reason,
            "elapsed_s": self.elapsed_s,
            "feedback_events": len(self.feedback),
        }


# ------------------------------------------------------------------- outcome

class AgenticCandidate(BaseModel):
    """One ranked candidate from the agent's final answer."""

    number: str
    title: Optional[str] = None
    why: str = ""
    confidence: float = 0.0
    passages: list[str] = Field(default_factory=list)


class AgenticSearchOutcome(BaseModel):
    """The full outcome of one :meth:`AgenticSearchRunner.run`."""

    search_type: str
    results: list[AgenticCandidate] = Field(default_factory=list)
    rationale: str = ""
    suggested_next_queries: list[str] = Field(default_factory=list)
    trace: SearchTrace
    stop_reason: str
    elapsed_s: float
    #: neutral-schema conversation; pass back as ``resume_messages`` to
    #: continue the SAME agent conversation with injected feedback
    messages: list[dict] = Field(default_factory=list)


# ------------------------------------------------------------- system prompts

_COMMON_RULES = """
Search method:
- Iterate query angles: terminology variants/synonyms, CPC art classes,
  component-level vs function-level phrasing, problem-oriented phrasing.
- Inspect the top hits (get_patent) before refining a query direction.
- Keep a working shortlist (the shortlist tool) as you find candidates.
- Stop when marginal novelty drops (new queries keep surfacing the same
  documents) or the budget is nearly spent.
- You MUST ALWAYS end by calling the finish tool with your ranked
  candidates, an overall rationale, and suggested next queries a human
  could try. Never end without calling finish.
"""

_INVALIDITY_SYSTEM = """You are an expert prior-art (invalidity) searcher \
working over an indexed patent corpus via tools.

Goal: find references that anticipate or render obvious the target claims.
Only documents effective BEFORE the prior-art cutoff date count; the tool
layer enforces the cutoff and the exclusion list (examiner-cited art, family
members, the target itself) automatically — do not waste queries on excluded
documents.
""" + _COMMON_RULES

_FTO_SYSTEM = """You are an expert freedom-to-operate (FTO) searcher working \
over an indexed patent corpus via tools.

Goal: find potentially in-force patents whose claims the described product
might practice. Favor patents within the in-force window given in the task;
results are leads for attorney review, not a clearance opinion — rank by how
closely the claims read on the product.
""" + _COMMON_RULES

_INFRINGEMENT_SYSTEM = """You are an expert infringement (evidence-of-use) \
analyst working via tools.

Goal: rank the candidate products listed in the task by how likely each
practices ALL of the target claim limitations, citing the evidence given.
You may use get_patent to re-read the target patent and the search tools for
context. In shortlist/finish, use the candidate PRODUCT NAME in the "number"
field. Results are leads for investigation, not infringement conclusions.
""" + _COMMON_RULES

_SYSTEM_PROMPTS = {
    "invalidity": _INVALIDITY_SYSTEM,
    "fto": _FTO_SYSTEM,
    "infringement": _INFRINGEMENT_SYSTEM,
}


# --------------------------------------------------------------- tool schemas

def _arr(description: str, item_type: str = "string") -> dict:
    return {"type": "array", "items": {"type": item_type}, "description": description}


_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": _arr("Search keywords/phrases (OR semantics with minimum_match)."),
        "required_keywords": _arr("Keywords that MUST all appear (AND semantics)."),
        "excluded_keywords": _arr("Tokens/phrases that must NOT appear."),
        "text": {"type": "string", "description": "Free-text query for phrase matching."},
        "minimum_match": {"type": "integer",
                          "description": "Minimum number of `keywords` that must match."},
        "fields": _arr("Fields to search; subset of title, abstract, claims, specification."),
        "art_classes": _arr("CPC/IPC art-class prefixes, e.g. ['G06F16', 'H04L']."),
        "inventors": _arr("Inventor-name substrings to require."),
        "assignees": _arr("Assignee-name substrings to require."),
        "before_date": {"type": "string",
                        "description": "Only documents effective before this date (YYYY-MM-DD)."},
        "after_date": {"type": "string",
                       "description": "Only documents effective after this date (YYYY-MM-DD)."},
        "countries": _arr("Country codes to allow, e.g. ['US', 'EP']."),
        "limit": {"type": "integer", "description": f"Max results (capped at {MAX_TOOL_RESULTS}).",
                  "default": 10},
    },
    "required": [],
}

_SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Natural-language query to embed and match."},
        "limit": {"type": "integer", "default": 10},
    },
    "required": ["text"],
}

_GET_PATENT_SCHEMA = {
    "type": "object",
    "properties": {"number": {"type": "string", "description": "Patent number, e.g. US7000001B1."}},
    "required": ["number"],
}

_SHORTLIST_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "description": "Your full current working candidate list (replaces the previous one).",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "why": {"type": "string"},
                    "key_passage": {"type": "string"},
                },
                "required": ["number", "why"],
            },
        },
    },
    "required": ["candidates"],
}

_FINISH_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "description": "Final ranked candidates, best first.",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "why": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "key_passages": _arr("Verbatim passages supporting relevance."),
                },
                "required": ["number", "why", "confidence"],
            },
        },
        "rationale": {"type": "string", "description": "Overall search rationale/summary."},
        "suggested_next_queries": _arr("Queries a human could try next."),
    },
    "required": ["candidates", "rationale"],
}


# ----------------------------------------------------------------- run state

class _RunState:
    """Mutable per-run state shared by the tool closures."""

    def __init__(self, exclusions: dict[str, list[str]], cutoff_date: Optional[date],
                 search_type: str):
        self.search_type = search_type
        self.cutoff_date = cutoff_date
        self.exclusions = exclusions
        self.excluded_numbers: list[PatentNumber] = []
        for numbers in exclusions.values():
            for raw in numbers:
                try:
                    self.excluded_numbers.append(PatentNumber.parse(raw))
                except ValueError:
                    logger.warning("Skipping unparseable exclusion number: %r", raw)
        #: identifiers the model has legitimately seen (patent numbers from
        #: search results; product names for infringement)
        self.seen: set[str] = set()
        self.queries: list[dict] = []
        self.shortlist: list[dict] = []
        self.shortlist_history: list[list[dict]] = []
        self.finish_payload: Optional[dict] = None

    def is_excluded(self, raw: str) -> bool:
        try:
            number = PatentNumber.parse(raw)
        except ValueError:
            return False
        return any(number.equivalent(e) for e in self.excluded_numbers)

    def mark_seen(self, raw: str) -> None:
        self.seen.add(self._seen_key(raw))

    def has_seen(self, raw: str) -> bool:
        return self._seen_key(raw) in self.seen

    @staticmethod
    def _seen_key(raw: str) -> str:
        try:
            number = PatentNumber.parse(raw)
            return f"{number.country_code}{number.number.lstrip('0') or '0'}"
        except ValueError:
            return raw.strip().lower()


def _reseed_seen(state: _RunState, messages: list[dict]) -> None:
    """Re-populate the seen-document set from a resumed conversation's
    search-tool results, so the agent may finish with candidates it found in
    earlier runs of the same conversation."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (block.get("type") == "tool_result"
                    and block.get("name") in ("search_patents", "semantic_search")):
                try:
                    payload = json.loads(block.get("content") or "")
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                for item in payload.get("results") or []:
                    number = item.get("number") if isinstance(item, dict) else None
                    if number and not state.is_excluded(str(number)):
                        state.mark_seen(str(number))


def _parse_date(raw: Any) -> Optional[date]:
    if not raw:
        return None
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _compact_result(result: SearchResult) -> dict:
    patent = result.patent
    effective = None
    if patent is not None:
        effective = patent.best_effective_date() or patent.publication_date
    return {
        "number": str(result.patent_number),
        "title": patent.title if patent else None,
        "date": effective.isoformat() if effective else None,
        "score": round(float(result.score), 4),
        "passages": [p.text[:SNIPPET_CHARS] for p in result.passages[:2]],
    }


def _compact_patent(patent: Patent) -> dict:
    claims: list[dict] = [
        {"number": c.number, "text": c.text[:1200]}
        for c in (patent.independent_claims or patent.claims)[:6]
    ]
    return {
        "number": str(patent.patent_number),
        "title": patent.title,
        "abstract": (patent.abstract or "")[:1000] or None,
        "independent_claims": claims,
        "cpc": patent.cpc_codes[:10],
        "priority_date": patent.priority_date.isoformat() if patent.priority_date else None,
        "filing_date": patent.filing_date.isoformat() if patent.filing_date else None,
        "publication_date": patent.publication_date.isoformat() if patent.publication_date else None,
    }


def _claims_text(patent: Optional[Patent], claims: Optional[Sequence] = None) -> str:
    """Render the selected claims' text (claim numbers or raw texts accepted)."""
    if claims and all(isinstance(c, str) for c in claims):
        return "\n".join(str(c) for c in claims)
    if patent is None:
        return ""
    selected: list[Claim]
    if claims:
        wanted = {int(c) for c in claims}
        selected = [c for c in patent.claims if c.number in wanted] or list(patent.claims)
    else:
        selected = patent.independent_claims or list(patent.claims)
    return "\n".join(f"Claim {c.number}: {c.text}" for c in selected)


# -------------------------------------------------------------------- runner

class AgenticSearchRunner:
    """Runs one LLM agent over the search toolbelt with hard tool-layer rules.

    Args:
        keyword_store: any :class:`~patentkit.search.base.KeywordStore`.
        vector_store: optional vector store; registers the ``semantic_search``
            tool when present.
        llm: an :class:`patentkit.llm.LLM` whose provider supports tool use.
        max_steps: maximum agent rounds per run.
        budget_seconds: wall-clock budget per run (default finishes ≤3 min).
    """

    def __init__(self, keyword_store: KeywordStore, vector_store: Optional[VectorStore] = None,
                 llm=None, max_steps: int = DEFAULT_MAX_STEPS,
                 budget_seconds: float = DEFAULT_BUDGET_SECONDS):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.llm = llm
        self.max_steps = max_steps
        self.budget_seconds = budget_seconds

    # ------------------------------------------------------------- toolbelt
    def _build_tools(self, state: _RunState) -> list[ToolDef]:
        runner = self

        def search_patents(args: dict) -> Any:
            query = SearchQuery(
                keywords=[str(k) for k in args.get("keywords") or []],
                required_keywords=[str(k) for k in args.get("required_keywords") or []],
                excluded_keywords=[str(k) for k in args.get("excluded_keywords") or []],
                text=args.get("text"),
                minimum_match=args.get("minimum_match"),
                fields=list(args.get("fields") or
                            ["title", "abstract", "claims", "specification"]),
                art_classes=[str(c) for c in args.get("art_classes") or []],
                inventors=[str(i) for i in args.get("inventors") or []],
                assignees=[str(a) for a in args.get("assignees") or []],
                before_date=_parse_date(args.get("before_date")),
                after_date=_parse_date(args.get("after_date")),
                countries=[str(c) for c in args.get("countries") or []],
                limit=max(1, min(int(args.get("limit") or 10), MAX_TOOL_RESULTS)),
            )
            # HARD ENFORCEMENT — exclusions always applied; invalidity cutoff clamped.
            query.exclude_numbers = list(state.excluded_numbers)
            if state.search_type == "invalidity" and state.cutoff_date is not None:
                query.before_date = (min(query.before_date, state.cutoff_date)
                                     if query.before_date else state.cutoff_date)
            state.queries.append({"tool": "search_patents",
                                  "arguments": {k: v for k, v in args.items() if v}})
            results = runner.keyword_store.search(query)
            results = [r for r in results if not state.is_excluded(str(r.patent_number))]
            for result in results:
                state.mark_seen(str(result.patent_number))
            return {"count": len(results), "results": [_compact_result(r) for r in results]}

        def semantic_search(args: dict) -> Any:
            text = str(args.get("text") or "").strip()
            if not text:
                return {"error": "semantic_search requires non-empty 'text'"}
            limit = max(1, min(int(args.get("limit") or 10), MAX_TOOL_RESULTS))
            query = SearchQuery(exclude_numbers=list(state.excluded_numbers))
            if state.search_type == "invalidity" and state.cutoff_date is not None:
                query.before_date = state.cutoff_date
            state.queries.append({"tool": "semantic_search",
                                  "arguments": {"text": text, "limit": limit}})
            results = runner.vector_store.search_text(text, limit=limit, query=query)
            results = [r for r in results if not state.is_excluded(str(r.patent_number))]
            for result in results:
                state.mark_seen(str(result.patent_number))
            return {"count": len(results), "results": [_compact_result(r) for r in results]}

        def get_patent(args: dict) -> Any:
            raw = str(args.get("number") or "")
            try:
                number = PatentNumber.parse(raw)
            except ValueError as exc:
                return {"error": str(exc)}
            patent = runner.keyword_store.get(number)
            if patent is None and runner.vector_store is not None:
                patent = runner.vector_store.get(number)
            if patent is None:
                return {"error": f"Patent {raw} not found in the configured stores."}
            return _compact_patent(patent)

        def _validate_candidates(raw_candidates: Any) -> tuple[list[dict], list[str]]:
            accepted: list[dict] = []
            rejected: list[str] = []
            for item in raw_candidates if isinstance(raw_candidates, list) else []:
                if not isinstance(item, dict) or not item.get("number"):
                    rejected.append(f"malformed candidate: {item!r}")
                    continue
                number = str(item["number"])
                if state.is_excluded(number):
                    rejected.append(f"{number}: excluded by the search exclusion list")
                    continue
                if not state.has_seen(number):
                    rejected.append(f"{number}: not seen in any search result of this run")
                    continue
                accepted.append(item)
            return accepted, rejected

        def shortlist(args: dict) -> Any:
            accepted, rejected = _validate_candidates(args.get("candidates"))
            state.shortlist = accepted
            state.shortlist_history.append([dict(c) for c in accepted])
            return {"accepted": len(accepted), "rejected": rejected}

        def finish(args: dict) -> Any:
            raw_candidates = args.get("candidates")
            if not isinstance(raw_candidates, list):
                return {"error": "finish requires a 'candidates' array"}
            accepted, rejected = _validate_candidates(raw_candidates)
            state.finish_payload = {
                "candidates": accepted,
                "rationale": str(args.get("rationale") or ""),
                "suggested_next_queries": [str(q) for q in
                                           args.get("suggested_next_queries") or []],
                "rejected": rejected,
            }
            return {"ok": True, "accepted": len(accepted), "rejected": rejected}

        tools = [
            ToolDef("search_patents",
                    "Keyword (BM25) search over the indexed patent corpus with the full "
                    "query parameter set. The run's exclusion list and prior-art cutoff "
                    "are enforced automatically.",
                    _SEARCH_SCHEMA, search_patents),
            ToolDef("get_patent",
                    "Fetch one patent's compact record: title, abstract, independent "
                    "claims, CPC codes, and dates.",
                    _GET_PATENT_SCHEMA, get_patent),
            ToolDef("shortlist",
                    "Record/update your full working candidate list (replaces the "
                    "previous shortlist). Only documents seen in this run's search "
                    "results are accepted.",
                    _SHORTLIST_SCHEMA, shortlist),
            ToolDef("finish",
                    "End the search with your final ranked candidates, overall "
                    "rationale, and suggested next queries. ALWAYS call this to finish.",
                    _FINISH_SCHEMA, finish),
        ]
        if self.vector_store is not None:
            tools.insert(1, ToolDef(
                "semantic_search",
                "Semantic (embedding) search over the corpus with a natural-language "
                "query. Exclusions and the prior-art cutoff are enforced automatically.",
                _SEMANTIC_SCHEMA, semantic_search))
        return tools

    # ------------------------------------------------------- task rendering
    @staticmethod
    def _exclusion_summary(exclusions: dict[str, list[str]]) -> str:
        if not exclusions:
            return "none"
        return "; ".join(f"{reason}: {len(numbers)} document(s)"
                         for reason, numbers in exclusions.items() if numbers) or "none"

    def _task_message(self, search_type: str, *, target_patent: Optional[Patent],
                      claims: Optional[Sequence], product_description: Optional[str],
                      evidence: Optional[Sequence], exclusions: dict[str, list[str]],
                      cutoff_date: Optional[date], final_k: int) -> str:
        lines: list[str] = []
        if search_type == "invalidity":
            assert target_patent is not None
            lines.append(f"Find prior art that invalidates {target_patent.patent_number}: "
                         f"{target_patent.title or '(untitled)'}")
            lines.append(f"Prior-art cutoff (enforced): documents effective before "
                         f"{cutoff_date.isoformat() if cutoff_date else 'n/a'}")
            lines.append(f"Exclusions (enforced): {self._exclusion_summary(exclusions)}")
            if target_patent.cpc_codes:
                lines.append("Target CPC codes: " + ", ".join(target_patent.cpc_codes[:8]))
            lines.append("Target claims:\n" + _claims_text(target_patent, claims))
        elif search_type == "fto":
            lines.append("Freedom-to-operate screen for this product:")
            lines.append(product_description or "(no description)")
            if cutoff_date:
                lines.append(f"In-force window: prefer patents filed after "
                             f"{cutoff_date.isoformat()} (possibly still in force).")
            if exclusions:
                lines.append(f"Exclusions (enforced): {self._exclusion_summary(exclusions)}")
        else:  # infringement
            assert target_patent is not None
            lines.append(f"Evidence-of-use ranking for {target_patent.patent_number}: "
                         f"{target_patent.title or '(untitled)'}")
            lines.append("Claim limitations:\n" + _claims_text(target_patent, claims))
            lines.append("Candidate products (use the product NAME as the candidate "
                         "'number' in shortlist/finish):")
            for i, product in enumerate(evidence or [], start=1):
                name = str(product.get("name", f"product {i}"))
                description = str(product.get("description", ""))[:600]
                ev = str(product.get("evidence", ""))[:400]
                line = f"{i}. {name} — {description}"
                if ev:
                    line += f" | evidence: {ev}"
                lines.append(line)
        lines.append(f"\nReturn up to {final_k} ranked candidates via the finish tool, "
                     "best first, each with a confidence (0-1), a one-sentence 'why', "
                     "and verbatim key passages.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ run
    def run(
        self,
        search_type: str,
        *,
        target_patent: Optional[Patent] = None,
        claims: Optional[Sequence] = None,
        product_description: Optional[str] = None,
        evidence: Optional[Sequence[dict]] = None,
        exclusions: dict[str, list[str]],
        cutoff_date: Optional[date] = None,
        final_k: int = 25,
        feedback_messages: Sequence[str] = (),
        resume_messages: Optional[list[dict]] = None,
        on_step: Optional[Callable[[TraceStep], None]] = None,
    ) -> AgenticSearchOutcome:
        """Run the agent and return the ranked outcome + trace + conversation.

        Args:
            search_type: "invalidity", "fto", or "infringement".
            target_patent: the target patent (invalidity / infringement).
            claims: claim numbers (resolved against ``target_patent``) or raw
                claim texts.
            product_description: the product (fto).
            evidence: candidate products for infringement —
                ``[{"name", "description", "evidence"?}]``.
            exclusions: reason -> patent numbers; ALWAYS enforced in the tool
                layer.
            cutoff_date: prior-art cutoff (invalidity, clamped in the tool
                layer) or in-force window start (fto, prompt-level).
            final_k: how many ranked candidates to ask for / return.
            feedback_messages: user feedback injected as user messages (used
                when resuming a conversation, or as pre-run guidance).
            resume_messages: a previous outcome's ``messages`` to continue
                the same agent conversation.
            on_step: live callback receiving every TraceStep.
        """
        if search_type not in _SYSTEM_PROMPTS:
            raise ValueError(f"Unknown search_type {search_type!r}")
        if self.llm is None:
            raise ValueError("AgenticSearchRunner requires an llm; use the search "
                             "agents' keys-free degraded mode instead.")

        state = _RunState(exclusions, cutoff_date, search_type)
        if search_type == "infringement":
            for product in evidence or []:
                name = str(product.get("name", "")).strip()
                if name:
                    state.mark_seen(name)
        tools = self._build_tools(state)

        if resume_messages:
            messages = [dict(m) for m in resume_messages]
            _reseed_seen(state, messages)
        else:
            messages = [user_text_message(self._task_message(
                search_type, target_patent=target_patent, claims=claims,
                product_description=product_description, evidence=evidence,
                exclusions=exclusions, cutoff_date=cutoff_date, final_k=final_k,
            ))]
        feedback_list = [str(f) for f in feedback_messages if str(f).strip()]
        for feedback in feedback_list:
            messages.append(user_text_message(
                "USER FEEDBACK on the search so far — adjust your queries and "
                "candidates accordingly:\n" + feedback))

        target = str(target_patent.patent_number) if target_patent else \
            (product_description or "")[:80] or None
        run = run_tool_loop(
            self.llm,
            system=_SYSTEM_PROMPTS[search_type],
            messages=messages,
            tools=tools,
            max_steps=self.max_steps,
            budget_seconds=self.budget_seconds,
            finish_tool="finish",
            on_step=on_step,
        )

        trace = SearchTrace(
            search_type=search_type,
            target=target,
            max_steps=self.max_steps,
            budget_seconds=self.budget_seconds,
            stop_reason=run.stop_reason,
            elapsed_s=run.elapsed_s,
            steps=[TraceStepModel.from_step(s) for s in run.steps],
            feedback=feedback_list,
            queries=list(state.queries),
            shortlist_history=list(state.shortlist_history),
            usage=dict(run.usage),
        )

        results, rationale, suggested = self._hydrate(state, search_type, final_k)
        return AgenticSearchOutcome(
            search_type=search_type,
            results=results,
            rationale=rationale,
            suggested_next_queries=suggested,
            trace=trace,
            stop_reason=run.stop_reason,
            elapsed_s=run.elapsed_s,
            messages=run.messages,
        )

    # ------------------------------------------------------------ hydration
    def _hydrate(self, state: _RunState, search_type: str,
                 final_k: int) -> tuple[list[AgenticCandidate], str, list[str]]:
        """Build ranked candidates from the finish payload (or the last
        shortlist when the run was truncated), hydrating titles from stores."""
        if state.finish_payload is not None:
            raw = state.finish_payload["candidates"]
            rationale = state.finish_payload["rationale"]
            suggested = state.finish_payload["suggested_next_queries"]
            default_confidence = None
        else:
            raw = state.shortlist
            rationale = ("Run ended before the agent called finish; returning its "
                         "last working shortlist.")
            suggested = []
            default_confidence = 0.5

        candidates: list[AgenticCandidate] = []
        for item in raw:
            number = str(item.get("number", ""))
            try:
                confidence = float(item.get("confidence", default_confidence
                                            if default_confidence is not None else 0.0))
            except (TypeError, ValueError):
                confidence = default_confidence or 0.0
            confidence = max(0.0, min(1.0, confidence))
            passages = [str(p) for p in item.get("key_passages") or []]
            if not passages and item.get("key_passage"):
                passages = [str(item["key_passage"])]
            title = None
            if search_type != "infringement":
                try:
                    patent = self.keyword_store.get(PatentNumber.parse(number))
                    if patent is None and self.vector_store is not None:
                        patent = self.vector_store.get(PatentNumber.parse(number))
                    if patent is not None:
                        title = patent.title
                        if not passages:
                            passages = [(patent.abstract or "")[:SNIPPET_CHARS]] \
                                if patent.abstract else []
                except ValueError:
                    pass
            candidates.append(AgenticCandidate(
                number=number, title=title, why=str(item.get("why", "")),
                confidence=confidence, passages=passages,
            ))
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:final_k], rationale, suggested


__all__ = [
    "AgenticCandidate",
    "AgenticSearchOutcome",
    "AgenticSearchRunner",
    "SearchTrace",
    "TraceStepModel",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_BUDGET_SECONDS",
]
