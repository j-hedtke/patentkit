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

# IPR-example querysets (from patentkit.evals toy IPR set; refs verified to
# scrape, predate the target's priority date, and not be examiner-cited).
QUERYSETS = [
    {
        "query_patent": "US5946647A",
        "claims": [1, 4, 6],
        "references": ["US5644735A", "US5859636A"],
        "metadata": {"proceeding": "IPR2020-00104", "toy": True},
    },
    {
        "query_patent": "US8046721B2",
        "claims": [1, 8],
        "references": ["US6209104B1", "US5821933A"],
        "metadata": {"proceeding": "IPR2020-00103", "toy": True},
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
