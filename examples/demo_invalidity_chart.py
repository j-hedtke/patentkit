"""patentkit demo — agentic invalidity search to a finished claim-chart DOCX.

One toy IPR-style queryset (Apple's '647 "data detectors" patent; the
dataset's proceeding labels are illustrative, not real PTAB citations),
end to end:

  1. corpus    — real patents: reuses data/eval_corpus/corpus.jsonl when
                 present, otherwise live-scrapes a small citation
                 neighborhood from Google Patents (resumable cache)
  2. search    — InvaliditySearchAgent's agentic tool-use loop over the
                 in-memory BM25 store (keys-free fallback is the degraded
                 keyword mode and is labeled as such)
  3. chart     — claim 1 split into atomic limitations; the top hit assessed
                 limitation-by-limitation with verbatim supporting quotes
  4. citations — each quote located in the reference's issued PDF and cited
                 by column/line ("col. 3, ll. 45-52")
  5. product   — color-coded DOCX claim chart

Run:
    python examples/demo_invalidity_chart.py      # outputs in data/demo/

Needs the pdf, docx, and scrape extras plus an ANTHROPIC_API_KEY or
OPENAI_API_KEY for the agentic search and chart analysis.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

import httpx

from patentkit.agents import InvaliditySearchAgent
from patentkit.analysis.invalidity import build_claim_chart
from patentkit.formatting.claim_chart import claim_chart_docx
from patentkit.models import Patent, PatentNumber
from patentkit.parsing.patent_pdf import format_patent_citation, locate_passage
from patentkit.search.bm25 import BM25Store

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

QUERY_PATENT = "US5946647A"  # toy IPR-style queryset (illustrative, not a real proceeding)
CLAIM_NUMBER = 1
GROUND_TRUTH = ["US5644735A", "US5859636A"]

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
        source = OUT_DIR / "demo_corpus.jsonl"
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


def download_reference_pdf(number: str) -> Path | None:
    """Fetch the issued-patent PDF that Google Patents links for ``number``."""
    pdf_path = OUT_DIR / f"{number}.pdf"
    if pdf_path.exists():
        return pdf_path
    headers = {"User-Agent": "Mozilla/5.0 (patentkit demo)"}
    page = httpx.get(
        f"https://patents.google.com/patent/{number}/en",
        headers=headers, follow_redirects=True, timeout=30,
    ).text
    match = re.search(r'<meta name="citation_pdf_url" content="([^"]+)"', page)
    if not match:
        return None
    pdf_path.write_bytes(
        httpx.get(match.group(1), headers=headers, follow_redirects=True, timeout=60).content
    )
    return pdf_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("1/5", f"corpus — toy IPR-style queryset {QUERY_PATENT}")
    patents = load_corpus()
    target = patents[QUERY_PATENT]
    print(f"target: {QUERY_PATENT} — {target.title}")
    print(f"ground-truth prior art (real references, toy queryset): {', '.join(GROUND_TRUTH)}")

    banner("2/5", "agentic invalidity search (model-driven tool-use loop, BM25 store)")
    store = BM25Store()
    store.index(patents.values())
    llm = pick_llm("medium")
    if llm is None:
        print("NOTE: no ANTHROPIC_API_KEY/OPENAI_API_KEY — running the DEGRADED")
        print("keyword-only fallback; this is NOT agentic-mode performance, and")
        print("the chart step needs an LLM, so the demo stops after the search.")
    agent = InvaliditySearchAgent(keyword_store=store, llm=llm)
    result = agent.search(target, claims=[CLAIM_NUMBER], final_k=5, budget_seconds=120)
    print(f"\nstop_reason={result.stop_reason}; top {len(result.results)} candidates:")
    for i, row in enumerate(result.results, 1):
        hit = " ◀ ground truth" if any(
            PatentNumber.parse(row["patent_number"]).equivalent(PatentNumber.parse(g))
            for g in GROUND_TRUTH) else ""
        print(f"  {i}. {row['patent_number']}  {row.get('title', '')[:60]}{hit}")
        if row.get("why"):
            print(f"     why: {row['why'][:120]}")
    if llm is None:
        sys.exit(0)

    top = result.results[0]["patent_number"]
    reference = patents[str(PatentNumber.parse(top))]

    banner("3/5", f"column/line locator — issued PDF for {top}")
    pdf_path = download_reference_pdf(top)
    locator = None
    if pdf_path:
        print(f"PDF: {pdf_path} — quotes will be fuzzy-located and cited by col./line")

        def locator(passage: str) -> str | None:
            loc = locate_passage(str(pdf_path), passage)
            return format_patent_citation(loc) if loc else None
    else:
        print("no PDF link found; chart will omit col./line citations")

    banner("4/5", f"claim chart — {QUERY_PATENT} claim {CLAIM_NUMBER} vs {top}")
    print("splitting claim into atomic limitations, assessing disclosure per")
    print("limitation with verbatim quotes (one HIGH-effort LLM call each)...")
    chart = build_claim_chart(
        target, CLAIM_NUMBER,
        references=[(top, reference.specification or reference.abstract or "")],
        llm=pick_llm("high"), locator=locator,
    )
    spotlight = next(
        (f for f in chart.references[0].findings if f.citation and f.quotes),
        chart.references[0].findings[0] if chart.references[0].findings else None,
    )
    if spotlight:
        print("\none element, fully traced:")
        print(f"  limitation: {spotlight.limitation.text}")
        print(f"  status:     {spotlight.status}")
        if spotlight.quotes:
            print(f'  quote:      "{spotlight.quotes[0][:160]}"')
        print(f"  citation:   {top}, {spotlight.citation or '(not located)'}")

    banner("5/5", "finished product — color-coded DOCX claim chart")
    (OUT_DIR / "chart.json").write_text(chart.model_dump_json(indent=2))
    docx_path = OUT_DIR / f"claim_chart_{QUERY_PATENT}_claim{CLAIM_NUMBER}.docx"
    claim_chart_docx(chart, str(docx_path))
    coverage = ", ".join(f"{n}: {f:.0%}" for n, f in chart.coverage_summary().items())
    print(f"wrote {docx_path}")
    print(f"coverage — {coverage}; combined: {chart.combined_coverage():.0%}")


if __name__ == "__main__":
    main()
