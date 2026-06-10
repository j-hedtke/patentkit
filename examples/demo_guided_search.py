"""patentkit demo — GUIDED agentic invalidity search (steerable, resumable).

One toy IPR-style queryset (Apple's '721 "slide to unlock" patent; the
dataset's proceeding labels are illustrative, not real PTAB citations),
showing the core differentiator: the search is ONE resumable agent
conversation the practitioner can steer with feedback between rounds.

  1. corpus    — real patents: reuses data/eval_corpus/corpus.jsonl when
                 present, otherwise live-scrapes a small citation
                 neighborhood from Google Patents (resumable cache)
  2. round 1   — InvaliditySearchAgent's agentic tool-use loop over the
                 in-memory BM25 store, on a modest budget; an unguided run
                 of this queryset found US5821933A but missed US6209104B1
  3. feedback  — a realistic practitioner steer, printed verbatim
  4. round 2   — the SAME conversation resumed (resume_messages=round 1's
                 conversation + feedback_messages) on a fresh small budget;
                 refined candidates and a delta vs round 1
  5. traces    — both reasoning traces saved to data/demo/

Run:
    python examples/demo_guided_search.py         # outputs in data/demo/

Needs an ANTHROPIC_API_KEY or OPENAI_API_KEY: guided search is a
conversation with the agent, so the keys-free degraded keyword mode has
nothing to resume — without a key the demo runs the (clearly labeled)
degraded single pass and stops.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from patentkit.agents import InvaliditySearchAgent
from patentkit.models import Patent, PatentNumber
from patentkit.search.bm25 import BM25Store

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

QUERY_PATENT = "US8046721B2"  # toy IPR-style queryset (illustrative, not a real proceeding)
CLAIM_NUMBER = 1
GROUND_TRUTH = ["US6209104B1", "US5821933A"]

FEEDBACK = (
    "Good start — US5821933A is on point. Now focus specifically on art "
    "about access control via a personal identification mechanism or "
    "predefined gesture/path entry on a touch-sensitive input; the "
    "petitioner also cited a patent about user identification by drawing "
    "a pattern. Avoid CAPTCHA-style art."
)

EVAL_CORPUS = Path("data/eval_corpus/corpus.jsonl")
OUT_DIR = Path("data/demo")
SCRAPE_SIZE = 40


def banner(step: str, text: str) -> None:
    print(f"\n━━━ {step} ━━━ {text}")


def pick_llm(effort: str):
    from patentkit.llm import get_llm

    if os.environ.get("ANTHROPIC_API_KEY"):
        return get_llm(effort, provider="anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        return get_llm(effort, provider="openai")
    return None


def load_corpus() -> dict[str, Patent]:
    """Real-patent corpus: eval corpus if built, else a small live scrape."""
    source = EVAL_CORPUS if EVAL_CORPUS.exists() else None
    if source is None:
        source = OUT_DIR / "demo_guided_corpus.jsonl"
        scrape_neighborhood(source)
    patents: dict[str, Patent] = {}
    with source.open() as fh:
        for line in fh:
            if line.strip():
                p = Patent.model_validate_json(line)
                patents[str(p.patent_number)] = p
    print(f"corpus: {len(patents)} real patents from {source}")
    return patents


def scrape_neighborhood(cache: Path) -> None:
    """Live-scrape target + ground truth + citation-graph distractors."""
    from patentkit.connectors.inference.google_patents import GooglePatentsScraper

    cache.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if cache.exists():
        have = {
            str(Patent.model_validate_json(line).patent_number)
            for line in cache.read_text().splitlines() if line.strip()
        }
    scraper = GooglePatentsScraper()

    def scrape(number: str) -> Patent | None:
        if str(PatentNumber.parse(number)) in have:
            return None
        time.sleep(0.4)
        try:
            patent = scraper.fetch(number)
        except Exception as exc:  # noqa: BLE001 - skip-and-continue scrape job
            print(f"  skip {number}: {exc}")
            return None
        with cache.open("a") as fh:
            fh.write(patent.model_dump_json() + "\n")
        have.add(str(patent.patent_number))
        return patent

    print(f"scraping ~{SCRAPE_SIZE} patents from Google Patents (one-time, cached)...")
    seeds = [scrape(n) for n in [QUERY_PATENT, *GROUND_TRUTH]]
    frontier: list[str] = []
    for seed in seeds:
        if seed is None:
            continue
        for cit in list(seed.citations) + list(seed.cited_by):
            pn = cit.patent_number
            if pn.country_code == "US" and not pn.number.startswith(("D", "RE")):
                frontier.append(str(pn))
    for number in dict.fromkeys(frontier):
        if len(have) >= SCRAPE_SIZE:
            break
        scrape(number)


def is_ground_truth(number: str) -> bool:
    pn = PatentNumber.parse(number)
    return any(pn.equivalent(PatentNumber.parse(g)) for g in GROUND_TRUTH)


def print_candidates(results: list[dict]) -> None:
    for i, row in enumerate(results, 1):
        hit = " ◀ ground truth" if is_ground_truth(row["patent_number"]) else ""
        print(f"  {i}. {row['patent_number']}  {row.get('title', '')[:60]}{hit}")
        if row.get("why"):
            print(f"     why: {row['why'][:120]}")


def found_set(results: list[dict]) -> set[str]:
    return {str(PatentNumber.parse(row["patent_number"])) for row in results}


def save_trace(result, path: Path) -> None:
    if result.trace is not None:
        path.write_text(result.trace.model_dump_json(indent=2))
        print(f"saved reasoning trace: {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("1/5", f"corpus — toy IPR-style queryset {QUERY_PATENT}")
    patents = load_corpus()
    target = patents[QUERY_PATENT]
    print(f"target: {QUERY_PATENT} — {target.title}")
    print(f"ground-truth prior art (real references, toy queryset): {', '.join(GROUND_TRUTH)}")
    print("(an UNGUIDED agentic run on this queryset found US5821933A but")
    print(" missed US6209104B1 — exactly the case practitioner steering fixes)")

    store = BM25Store()
    store.index(patents.values())
    llm = pick_llm("medium")
    agent = InvaliditySearchAgent(keyword_store=store, llm=llm)

    banner("2/5", "round 1 — initial agentic search (modest budget: 90s / 10 steps)")
    if llm is None:
        print("NOTE: no ANTHROPIC_API_KEY/OPENAI_API_KEY — running the DEGRADED")
        print("keyword-only fallback; this is NOT agentic-mode performance.")
    round1 = agent.search(target, claims=[CLAIM_NUMBER], final_k=5,
                          budget_seconds=90, max_steps=10)
    print(f"\nstop_reason={round1.stop_reason}; top {len(round1.results)} candidates:")
    print_candidates(round1.results)
    save_trace(round1, OUT_DIR / "trace_guided_round1.json")
    if llm is None:
        print("\nGuided search is a conversation with the agent — there is no")
        print("conversation to resume in degraded mode, so it requires an LLM.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY and rerun. Exiting.")
        sys.exit(0)

    banner("3/5", "practitioner feedback — steering the SAME conversation")
    print(f'feedback: "{FEEDBACK}"')

    banner("4/5", "round 2 — resume the agent with feedback (fresh budget: 60s / 8 steps)")
    round2 = agent.search(
        target, claims=[CLAIM_NUMBER], final_k=5,
        budget_seconds=60, max_steps=8,
        resume_messages=round1.conversation,
        feedback_messages=[FEEDBACK],
    )
    print(f"\nstop_reason={round2.stop_reason}; top {len(round2.results)} candidates:")
    print_candidates(round2.results)

    before, after = found_set(round1.results), found_set(round2.results)
    newly = sorted(after - before)
    dropped = sorted(before - after)
    print(f"\ndelta vs round 1 — newly found: {', '.join(newly) or '(none)'}; "
          f"dropped: {', '.join(dropped) or '(none)'}")
    missed = [g for g in GROUND_TRUTH if str(PatentNumber.parse(g)) not in after]
    if not missed:
        print("both ground-truth references are now in the top candidates ◀")
    else:
        print(f"still missing ground truth: {', '.join(missed)}")

    banner("5/5", "saved artifacts — both reasoning traces in data/demo/")
    save_trace(round2, OUT_DIR / "trace_guided_round2.json")
    print(f"round 1 trace: {OUT_DIR / 'trace_guided_round1.json'}")
    print(f"round 2 trace: {OUT_DIR / 'trace_guided_round2.json'}")
    print("each trace records every query the agent issued, every tool result,")
    print("the injected feedback, and the shortlist evolution — round 2's trace")
    print("shows the conversation continuing after the steer, not starting over.")


if __name__ == "__main__":
    main()
