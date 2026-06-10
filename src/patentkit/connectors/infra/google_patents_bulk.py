"""Resumable bulk scraping of Google Patents.

Wraps :class:`~patentkit.connectors.inference.google_patents.GooglePatentsScraper`
in a checkpointed batch job that writes one canonical
:class:`~patentkit.models.patent.Patent` per line (JSONL). Designed for
large id lists: failures are collected and skipped (retry them later from
``tracker.failed``), progress is checkpointed after every batch, and a
polite per-request minimum interval keeps the scraper rate well below
anything that would trip Google's defenses.

Purely synchronous by design — bulk scraping should be slow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Protocol, Union

from patentkit.connectors.infra.progress import FileProgressTracker
from patentkit.models.patent import Patent

logger = logging.getLogger(__name__)


class _Scraper(Protocol):
    def fetch(self, number: str) -> Patent: ...


@dataclass
class BulkScrapeStats:
    """Summary of one :meth:`BulkScrapeJob.run`."""

    requested: int = 0
    scraped: int = 0
    failed: int = 0
    skipped: int = 0
    out_path: Optional[str] = None
    failed_ids: list[str] = field(default_factory=list)


def _batched(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class BulkScrapeJob:
    """Scrape many patent numbers to a JSONL file, resumably.

    Args:
        numbers: patent numbers to scrape, in a stable order (resume relies
            on the same order being supplied on restart).
        out_path: JSONL output, appended to on resume.
        tracker: optional :class:`FileProgressTracker`; pass the same path on
            restart to resume after the last successfully scraped number.
        batch_size: checkpoint frequency.
        min_interval_s: polite minimum delay between page fetches.
        scraper: injectable scraper (anything with ``fetch(number) -> Patent``);
            defaults to a :class:`GooglePatentsScraper`.
    """

    def __init__(
        self,
        numbers: Iterable[str],
        out_path: Union[str, Path],
        tracker: Optional[FileProgressTracker] = None,
        batch_size: int = 50,
        min_interval_s: float = 0.5,
        scraper: Optional[_Scraper] = None,
    ):
        self.numbers = numbers
        self.out_path = Path(out_path)
        self.tracker = tracker
        self.batch_size = batch_size
        if scraper is None:
            from patentkit.connectors.inference.google_patents import (
                GooglePatentsScraper,
            )

            scraper = GooglePatentsScraper(min_interval_s=min_interval_s)
        self.scraper = scraper

    def run(self) -> BulkScrapeStats:
        stats = BulkScrapeStats(out_path=str(self.out_path))
        resume_after = self.tracker.resume_after() if self.tracker else None
        skipping = resume_after is not None
        if skipping:
            logger.info("Resuming bulk scrape after %s", resume_after)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as out:
            for batch in _batched(self.numbers, self.batch_size):
                for number in batch:
                    stats.requested += 1
                    if skipping:
                        stats.skipped += 1
                        if number == resume_after:
                            skipping = False
                        continue
                    try:
                        patent = self.scraper.fetch(number)
                    except Exception as exc:
                        logger.warning("Failed to scrape %s: %s", number, exc)
                        stats.failed += 1
                        stats.failed_ids.append(number)
                        if self.tracker:
                            self.tracker.record_failure(number)
                        continue
                    out.write(patent.model_dump_json() + "\n")
                    stats.scraped += 1
                    if self.tracker:
                        self.tracker.record_success(number)
                out.flush()
                if self.tracker:
                    self.tracker.save()
        if self.tracker:
            self.tracker.save()
        logger.info(
            "Bulk scrape done: %d scraped, %d failed, %d skipped (of %d)",
            stats.scraped, stats.failed, stats.skipped, stats.requested,
        )
        return stats
