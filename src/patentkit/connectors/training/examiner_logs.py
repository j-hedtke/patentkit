"""Examiner search-query log extraction (training data for search models).

USPTO examiners attach their search history ("Search information including
search strategy and results", document code ``SRNT``) to the file wrapper.
These PDFs contain EAST/PatFT query logs — real expert search queries for
the application's claims, ideal supervision for prior-art search models.

Pipeline (per granted patent): patent number -> application number (ODP
search) -> SRNT documents -> PDF text -> :func:`parse_search_queries`
(regex heuristics over EAST query syntax) -> JSONL of
:class:`ExaminerQueryRecord`. Resumable via
:class:`~patentkit.connectors.infra.progress.FileProgressTracker`.

Requires a USPTO ODP key (``USPTO_ODP_API_KEY``) via the injected
:class:`~patentkit.connectors.inference.file_wrapper.FileWrapperClient`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from pydantic import BaseModel, Field

from patentkit.connectors.inference.file_wrapper import (
    FileWrapperClient,
    PdfTextExtractor,
)
from patentkit.connectors.infra.progress import FileProgressTracker

logger = logging.getLogger(__name__)


class ExaminerQueryRecord(BaseModel):
    """The examiner search queries recovered for one patent."""

    patent_number: str
    application_number: str
    queries: list[str] = Field(default_factory=list)
    references_cited: list[str] = Field(default_factory=list)
    source_doc_date: Optional[str] = None


#: EAST/PatFT field suffixes like .clm. / .ab,ti. / .ccls.
_FIELD_SUFFIX_RE = re.compile(
    r"\.\s*(?:clm|clms|dclm|ab|abs|ti|spec|ccls|cpc|ipc|in|as|kwic)"
    r"(?:\s*,\s*(?:clm|clms|dclm|ab|abs|ti|spec|ccls|cpc|ipc|in|as|kwic))*\s*\.",
    re.IGNORECASE,
)
_BOOLEAN_RE = re.compile(r"\b(?:and|or|not|adj\d*|near\d*|same|with)\b", re.IGNORECASE)
#: line prefixes of EAST search-history tables: "L1 423 <query> ..." / "S3 ..."
_LINE_PREFIX_RE = re.compile(r"^[LS]\d+\b[\s|:]*(?:[\d,]+[\s|]+)?(?P<query>.+)$")
#: trailing table columns (databases, operators, timestamps) to strip
_TRAILER_RE = re.compile(
    r"\s+(?:US-PGPUB|USPAT|USOCR|FPRS|EPO|JPO|DERWENT|IBM[_-]TDB)\b.*$",
    re.IGNORECASE,
)


def _looks_like_query(candidate: str) -> bool:
    if _FIELD_SUFFIX_RE.search(candidate):
        return True
    has_boolean = bool(_BOOLEAN_RE.search(candidate))
    has_structure = (
        ('"' in candidate)
        or ("(" in candidate and ")" in candidate)
        or ("$" in candidate)
    )
    return has_boolean and has_structure


def parse_search_queries(text: str) -> list[str]:
    """Extract examiner query strings from SRNT search-report text.

    Heuristics target EAST/PatFT search-history syntax: numbered log lines
    (``L1 423 ("widget" and coupler).clm. USPAT ...`` / ``S3 ...``), field
    suffixes like ``.clm.``/``.ab,ti.``, and quoted/parenthesized boolean
    expressions. Trailing table columns (database names, timestamps) are
    stripped; duplicates are removed preserving first-seen order.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LINE_PREFIX_RE.match(line)
        candidate = match.group("query").strip() if match else line
        candidate = _TRAILER_RE.sub("", candidate).strip()
        if not candidate or not _looks_like_query(candidate):
            continue
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            queries.append(candidate)
    return queries


@dataclass
class ExaminerLogBuildStats:
    """Summary of one :meth:`ExaminerLogBuilder.build_for_patents` run."""

    requested: int = 0
    written: int = 0
    failed: int = 0
    skipped: int = 0
    failed_ids: list[str] = field(default_factory=list)


class ExaminerLogBuilder:
    """Builds a JSONL dataset of examiner queries for a list of patents."""

    def __init__(
        self,
        file_wrapper: FileWrapperClient,
        pdf_text_extractor: Optional[PdfTextExtractor] = None,
    ):
        self.file_wrapper = file_wrapper
        self.pdf_text_extractor = pdf_text_extractor

    def build_for_patents(
        self,
        numbers: Iterable[str],
        out_path: Union[str, Path],
        tracker: Optional[FileProgressTracker] = None,
    ) -> ExaminerLogBuildStats:
        """Fetch SRNT docs for each patent and append records to ``out_path``.

        Pass a tracker pointing at the same checkpoint file to resume an
        interrupted run.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stats = ExaminerLogBuildStats()
        resume_after = tracker.resume_after() if tracker else None
        skipping = resume_after is not None

        with out_path.open("a", encoding="utf-8") as out:
            for number in numbers:
                stats.requested += 1
                if skipping:
                    stats.skipped += 1
                    if number == resume_after:
                        skipping = False
                    continue
                try:
                    record = self._build_record(number)
                except Exception as exc:
                    logger.warning("Examiner log build failed for %s: %s", number, exc)
                    record = None
                if record is None:
                    stats.failed += 1
                    stats.failed_ids.append(number)
                    if tracker:
                        tracker.record_failure(number)
                else:
                    out.write(record.model_dump_json() + "\n")
                    out.flush()
                    stats.written += 1
                    if tracker:
                        tracker.record_success(number)
                if tracker:
                    tracker.save()
        return stats

    def _build_record(self, patent_number: str) -> Optional[ExaminerQueryRecord]:
        app_number = self.file_wrapper.app_number_for_patent(patent_number)
        if not app_number:
            logger.info("No application number found for %s", patent_number)
            return None
        srnt_docs = self.file_wrapper.get_documents_by_codes(app_number, ("SRNT",))
        if not srnt_docs:
            logger.info("No SRNT documents for %s (app %s)", patent_number, app_number)
            return None

        queries: list[str] = []
        seen: set[str] = set()
        source_doc_date: Optional[str] = None
        for doc in srnt_docs:
            source_doc_date = doc.date or source_doc_date
            for url in doc.pdf_urls[:1]:
                pdf_bytes = self.file_wrapper.download_pdf(url)
                extractor = self.pdf_text_extractor
                if extractor is None:
                    from patentkit.connectors.inference.file_wrapper import (
                        default_pdf_text_extractor,
                    )

                    extractor = default_pdf_text_extractor
                for query in parse_search_queries(extractor(pdf_bytes)):
                    if query.lower() not in seen:
                        seen.add(query.lower())
                        queries.append(query)
        if not queries:
            return None

        try:
            references = [
                str(c.patent_number)
                for c in self.file_wrapper.get_examiner_cited_art(app_number)
            ]
        except Exception as exc:
            logger.debug("Citation lookup failed for %s: %s", app_number, exc)
            references = []

        return ExaminerQueryRecord(
            patent_number=patent_number,
            application_number=app_number,
            queries=queries,
            references_cited=references,
            source_doc_date=source_doc_date,
        )
