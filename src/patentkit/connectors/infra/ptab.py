"""PTAB (Patent Trial and Appeal Board) proceedings API client.

Data source: https://developer.uspto.gov/ptab-api — public, no auth.
Covers AIA trials including inter partes reviews (IPRs): proceeding
metadata, the full document list per proceeding (petitions, institution
decisions, final written decisions), and document downloads.

Useful for invalidity work: IPR final written decisions record which prior
art combinations actually invalidated (or failed to invalidate) claims.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from pydantic import BaseModel, Field

from patentkit.connectors.http import RateLimiter, download, request_json

logger = logging.getLogger(__name__)

PTAB_BASE_URL = "https://developer.uspto.gov/ptab-api"

#: proceeding status categories indicating a final decision was reached
FINAL_STATUS_CATEGORIES: frozenset[str] = frozenset(
    {"FWD Entered", "Final Written Decision"}
)


class IprProceeding(BaseModel):
    """One PTAB proceeding (IPR), with the raw API record attached."""

    proceeding_number: str
    patent_number: Optional[str] = None
    status: Optional[str] = None
    filing_date: Optional[str] = None
    petitioner: Optional[str] = None
    patent_owner: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, record: dict[str, Any]) -> "IprProceeding":
        return cls(
            proceeding_number=str(record.get("proceedingNumber", "")),
            patent_number=record.get("respondentPatentNumber"),
            status=record.get("proceedingStatusCategory"),
            filing_date=record.get("proceedingFilingDate")
            or record.get("accordedFilingDate"),
            petitioner=record.get("petitionerPartyName"),
            patent_owner=record.get("respondentPartyName")
            or record.get("patentOwnerName"),
            raw=record,
        )


class IprDocument(BaseModel):
    """One document filed in a PTAB proceeding."""

    document_id: str
    title: Optional[str] = None
    type_name: Optional[str] = None
    category: Optional[str] = None
    filing_date: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, record: dict[str, Any]) -> "IprDocument":
        return cls(
            document_id=str(record.get("documentIdentifier", "")),
            title=record.get("documentTitleText"),
            type_name=record.get("documentTypeName"),
            category=record.get("documentCategory"),
            filing_date=record.get("documentFilingDate"),
            raw=record,
        )

    @property
    def is_final_decision(self) -> bool:
        """Heuristic match for final written decision documents."""
        category = (self.category or "").lower()
        type_name = (self.type_name or "").lower()
        title = (self.title or "").lower()
        return (
            category == "final"
            or type_name == "final decision"
            or "final written decision" in title
            or title == "termination decision document"
        )


class PtabClient:
    """Client for the public PTAB API (no auth required)."""

    def __init__(self, *, min_interval_s: float = 1.0, timeout: float = 60.0):
        self._rate_limiter = RateLimiter(min_interval_s)
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return request_json(
            "GET",
            f"{PTAB_BASE_URL}/{path}",
            params=params,
            timeout=self.timeout,
            rate_limiter=self._rate_limiter,
        )

    def iter_ipr_proceedings(
        self,
        filed_from: Optional[str] = None,
        filed_to: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Iterator[IprProceeding]:
        """Paginated generator over IPR proceedings.

        Args:
            filed_from / filed_to: ``YYYY-MM-DD`` proceeding-filing-date bounds.
            status: optional ``proceedingStatusCategory`` filter (e.g.
                ``"FWD Entered"``); applied client-side.
        """
        params: dict[str, Any] = {"subproceedingTypeCategory": "IPR"}
        if filed_from:
            params["proceedingFilingFromDate"] = filed_from
        if filed_to:
            params["proceedingFilingToDate"] = filed_to

        consumed = 0
        total: Optional[int] = None
        while total is None or consumed < total:
            page = self._get(
                "proceedings", {**params, "recordStartNumber": consumed}
            )
            total = int(page.get("recordTotalQuantity", 0))
            results = page.get("results", []) or []
            if not results:
                break
            for record in results:
                consumed += 1
                if status and record.get("proceedingStatusCategory") != status:
                    continue
                yield IprProceeding.from_api(record)

    def list_proceeding_documents(self, proceeding_number: str) -> list[IprDocument]:
        """All documents filed in one proceeding."""
        documents: list[IprDocument] = []
        consumed = 0
        total: Optional[int] = None
        while total is None or consumed < total:
            page = self._get(
                "documents",
                {
                    "proceedingNumber": proceeding_number,
                    "recordStartNumber": consumed,
                },
            )
            total = int(page.get("recordTotalQuantity", 0))
            results = page.get("results", []) or []
            if not results:
                break
            consumed += len(results)
            documents.extend(IprDocument.from_api(r) for r in results)
        return documents

    def download_document(
        self, doc_id: str, dest: Optional[Union[str, Path]] = None
    ) -> Union[bytes, Path]:
        """Download one document (PDF). Returns bytes, or the Path if ``dest``."""
        return download(
            f"{PTAB_BASE_URL}/documents/{doc_id}/download",
            dest,
            rate_limiter=self._rate_limiter,
            timeout=self.timeout,
        )
