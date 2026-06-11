"""Build a real eval corpus by scraping Google Patents.

Seeds from IPR-example querysets (target patent + ground-truth prior art),
then fills the corpus with topically-related distractors discovered by
breadth-first traversal of the citation graph (citations + cited-by of the
seeds), exactly how the production system sourced neighborhoods. Output is
canonical-Patent JSONL plus a manifest, resumable on re-run.

Usage:
    python scripts/build_eval_corpus.py [--size 300] [--out data/eval_corpus]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

from patentkit.models import Patent, PatentNumber
from patentkit.connectors.inference.google_patents import GooglePatentsScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_eval_corpus")

# Real IPR queryset: IPR2020-01018, Unified Patents, LLC v. Voice Tech Corp.
# Final Written Decision 2021-12-13 held claims 1-8 of US10491679B2 obvious
# over Wong (US20060235700A1) in view of Beauregard (US6438545B1); affirmed,
# Voice Tech Corp. v. Unified Patents, LLC, No. 22-2163 (Fed. Cir. Aug. 1,
# 2024) (precedential). Ground truth = the references of the winning ground.
QUERYSETS = [
    {
        "query_patent": "US10491679B2",
        "claims": [1],
        "references": ["US20060235700A1", "US6438545B1"],
        "metadata": {
            "proceeding": "IPR2020-01018",
            "toy": False,
            "outcome": "claims 1-8 unpatentable (103) over Wong in view of Beauregard",
            "fwd_date": "2021-12-13",
            "affirmed": "Fed. Cir. No. 22-2163 (2024-08-01, precedential)",
        },
    },
]

MIN_INTERVAL_S = 0.4


def load_existing(corpus_path: Path) -> dict[str, Patent]:
    patents: dict[str, Patent] = {}
    if corpus_path.exists():
        with corpus_path.open() as fh:
            for line in fh:
                if line.strip():
                    p = Patent.model_validate_json(line)
                    patents[str(p.patent_number)] = p
    return patents


def us_neighbor_numbers(patent: Patent) -> list[str]:
    """US citation + cited-by numbers of a patent, deduped, parseable only."""
    out = []
    for cit in list(patent.citations) + list(patent.cited_by):
        pn = cit.patent_number
        if pn.country_code == "US" and not pn.number.startswith(("D", "RE")):
            out.append(str(pn))
    return list(dict.fromkeys(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--out", default="data/eval_corpus")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    manifest_path = out_dir / "manifest.json"

    scraper = GooglePatentsScraper()
    patents = load_existing(corpus_path)
    log.info("resuming with %d already-scraped patents", len(patents))

    failed: set[str] = set()

    def scrape(number: str) -> Patent | None:
        if number in patents:
            return patents[number]
        if number in failed:
            return None
        time.sleep(MIN_INTERVAL_S)
        try:
            patent = scraper.fetch(number)
        except Exception as exc:  # noqa: BLE001 - skip & continue is the job's contract
            log.warning("failed %s: %s", number, exc)
            failed.add(number)
            return None
        patents[str(patent.patent_number)] = patent
        with corpus_path.open("a") as fh:
            fh.write(patent.model_dump_json() + "\n")
        return patent

    # 1. Mandatory: targets + ground-truth references.
    seeds: list[Patent] = []
    for qs in QUERYSETS:
        for number in [qs["query_patent"], *qs["references"]]:
            patent = scrape(number)
            if patent is None and number not in patents:
                raise SystemExit(f"mandatory seed {number} failed to scrape")
            seeds.append(patents[PatentNumber.parse(number).__str__()]
                         if str(PatentNumber.parse(number)) in patents else patent)
    log.info("scraped %d mandatory seeds", len(seeds))

    # 2. Distractor frontier: interleave citation neighborhoods of all seeds,
    #    deterministic shuffle so re-runs are stable.
    frontier: list[str] = []
    for seed in seeds:
        frontier.extend(us_neighbor_numbers(seed))
    rng = random.Random(46647)
    frontier = list(dict.fromkeys(frontier))
    rng.shuffle(frontier)
    log.info("frontier of %d candidate distractors", len(frontier))

    for number in frontier:
        if len(patents) >= args.size:
            break
        scrape(number)
        if len(patents) % 25 == 0:
            log.info("corpus at %d/%d (failed: %d)", len(patents), args.size, len(failed))

    manifest = {
        "querysets": QUERYSETS,
        "corpus_size": len(patents),
        "failed_count": len(failed),
        "source": "patents.google.com (live scrape, canonical patentkit model)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("DONE: %d patents in %s", len(patents), corpus_path)


if __name__ == "__main__":
    main()
