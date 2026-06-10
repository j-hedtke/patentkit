"""USPTO Open Data Portal (ODP) file-wrapper client.

Data source: https://api.uspto.gov/api/v1/patent/applications/ — application
metadata, continuity, and the full prosecution-history document bag (office
actions, notices of allowance, applicant remarks, examiner search notes, ...).

Auth: an ODP API key sent in the ``X-API-KEY`` header. Get one at
https://data.uspto.gov/apis/getting-started and set ``USPTO_ODP_API_KEY`` (or
pass ``api_key=...``).

Common document codes:

- ``CTNF`` — non-final rejection (office action)
- ``CTFR`` — final rejection
- ``NOA``  — notice of allowance
- ``REM``  — applicant remarks/arguments
- ``SPEC`` — specification as filed
- ``SRNT`` — examiner search notes / search report (EAST query logs)

PDF text extraction is injectable; the default uses PyMuPDF if installed
(``pip install patentkit[pdf]``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from pydantic import BaseModel, Field

from patentkit.config import resolve_key
from patentkit.connectors.http import RateLimiter, download, request_json
from patentkit.models.patent import Citation, Patent, PatentNumber, SourceRecord

logger = logging.getLogger(__name__)

ODP_BASE_URL = "https://api.uspto.gov/api/v1/patent/applications"

#: callable taking raw PDF bytes and returning extracted text
PdfTextExtractor = Callable[[bytes], str]

DEFAULT_FILE_WRAPPER_CODES: tuple[str, ...] = ("CTNF", "NOA", "REM")


def default_pdf_text_extractor(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (lazy import)."""
    try:
        try:
            import pymupdf as fitz  # modern import name
        except ImportError:
            import fitz  # classic import name
    except ImportError as exc:
        raise ImportError(
            "PDF text extraction requires PyMuPDF. Install it with "
            "`pip install patentkit[pdf]` (or `pip install pymupdf`), or pass "
            "your own pdf_text_extractor=... callable."
        ) from exc
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


class FileWrapperDocument(BaseModel):
    """One entry of an application's ODP document bag."""

    code: str
    description: Optional[str] = None
    date: Optional[str] = None  # official date as reported by ODP (ISO-ish)
    pdf_urls: list[str] = Field(default_factory=list)


class FileWrapperClient:
    """Client for the USPTO ODP patent-applications API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        min_interval_s: float = 0.0,
        timeout: float = 30.0,
        pdf_text_extractor: Optional[PdfTextExtractor] = None,
    ):
        self.api_key = resolve_key("USPTO_ODP_API_KEY", api_key)
        self.timeout = timeout
        self._rate_limiter = RateLimiter(min_interval_s)
        self._pdf_text_extractor = pdf_text_extractor

    def _headers(self) -> dict[str, str]:
        return {"X-API-KEY": self.api_key}

    def _get_json(self, path: str) -> dict[str, Any]:
        return request_json(
            "GET",
            f"{ODP_BASE_URL}/{path}",
            headers=self._headers(),
            timeout=self.timeout,
            rate_limiter=self._rate_limiter,
        )

    ## Application metadata ###############################################

    def get_application(self, app_number: str) -> dict[str, Any]:
        """Fetch raw application data (``patentFileWrapperDataBag``)."""
        return self._get_json(app_number)

    def get_continuity(self, app_number: str) -> tuple[list[str], list[str]]:
        """Return (parent application numbers, child application numbers)."""
        data = self._get_json(f"{app_number}/continuity")
        wrapper = (data.get("patentFileWrapperDataBag") or [{}])[0]
        parents = [
            p["parentApplicationNumberText"]
            for p in wrapper.get("parentContinuityBag", [])
            if p.get("parentApplicationNumberText")
        ]
        children = [
            c["childApplicationNumberText"]
            for c in wrapper.get("childContinuityBag", [])
            if c.get("childApplicationNumberText")
        ]
        return parents, children

    ## Documents ##########################################################

    def list_documents(self, app_number: str) -> list[FileWrapperDocument]:
        """List the application's document bag with PDF download urls."""
        data = self._get_json(f"{app_number}/documents")
        documents: list[FileWrapperDocument] = []
        for doc in data.get("documentBag", []) or []:
            pdf_urls = [
                option["downloadUrl"]
                for option in doc.get("downloadOptionBag", []) or []
                if option.get("mimeTypeIdentifier") == "PDF" and option.get("downloadUrl")
            ]
            documents.append(
                FileWrapperDocument(
                    code=doc.get("documentCode", ""),
                    description=doc.get("documentCodeDescriptionText")
                    or doc.get("documentDescription"),
                    date=doc.get("officialDate"),
                    pdf_urls=pdf_urls,
                )
            )
        return documents

    def get_documents_by_codes(
        self, app_number: str, codes: Iterable[str]
    ) -> list[FileWrapperDocument]:
        """Documents matching any of ``codes``, oldest first."""
        wanted = set(codes)
        matched = [d for d in self.list_documents(app_number) if d.code in wanted]
        matched.sort(key=lambda d: d.date or "")
        return matched

    def download_pdf(self, url: str) -> bytes:
        """Download one document PDF (ODP download urls require the API key)."""
        data = download(
            url, headers=self._headers(), rate_limiter=self._rate_limiter
        )
        assert isinstance(data, bytes)
        return data

    ## Search #############################################################

    def search_applications(
        self,
        patent_number: Optional[Union[str, PatentNumber]] = None,
        grant_date_range: Optional[tuple[str, str]] = None,
        *,
        query: str = "",
        filters: Optional[list[dict[str, Any]]] = None,
        offset: int = 0,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """POST /search. Returns the raw result-bag entries.

        ``grant_date_range`` is an inclusive ``("YYYY-MM-DD", "YYYY-MM-DD")``
        pair translated to a ``grantDate:[from TO to]`` query.
        """
        q = query
        body_filters = list(filters or [])
        if patent_number is not None:
            number = (
                patent_number.number
                if isinstance(patent_number, PatentNumber)
                else PatentNumber.parse(str(patent_number)).number
            )
            body_filters.append({"name": "patentNumber", "value": [number]})
        if grant_date_range is not None:
            start, end = grant_date_range
            range_q = f"grantDate:[{start} TO {end}]"
            q = f"{q} AND {range_q}" if q else range_q
        body: dict[str, Any] = {
            "q": q,
            "filters": body_filters,
            "pagination": {"offset": offset, "limit": limit},
        }
        data = request_json(
            "POST",
            f"{ODP_BASE_URL}/search",
            headers=self._headers(),
            json=body,
            timeout=self.timeout,
            rate_limiter=self._rate_limiter,
        )
        for key in ("patentBag", "patentFileWrapperDataBag", "results"):
            if key in data:
                return data[key] or []
        return []

    def app_number_for_patent(
        self, patent_number: Union[str, PatentNumber]
    ) -> Optional[str]:
        """Look up the application number behind a granted patent number."""
        for entry in self.search_applications(patent_number=patent_number, limit=5):
            app = (
                entry.get("applicationNumberText")
                or (entry.get("applicationMetaData") or {}).get("applicationNumberText")
                or entry.get("applicationNumber")
            )
            if app:
                return str(app)
        return None

    ## Derived content ####################################################

    def get_examiner_cited_art(self, app_number: str) -> list[Citation]:
        """Best-effort parse of cited art from the ODP application record.

        ODP exposes citations inconsistently across records; this scans the
        known citation bags and flags examiner-cited entries. Returns an
        empty list when no citation data is present.
        """
        data = self.get_application(app_number)
        wrapper = (data.get("patentFileWrapperDataBag") or [{}])[0]
        citations: list[Citation] = []
        for bag_key in ("citationBag", "referenceCitationBag", "patentCitationBag"):
            for entry in wrapper.get(bag_key, []) or []:
                raw_number = (
                    entry.get("citedPatentNumber")
                    or entry.get("patentNumber")
                    or entry.get("publicationNumber")
                    or entry.get("citationDocumentIdentifier")
                )
                if not raw_number:
                    continue
                try:
                    pn = PatentNumber.parse(str(raw_number))
                except ValueError:
                    continue
                category = str(
                    entry.get("citationCategoryCode")
                    or entry.get("citedBy")
                    or entry.get("category")
                    or ""
                ).lower()
                is_examiner = "examiner" in category or entry.get(
                    "examinerCitedReferenceIndicator"
                ) in (True, "true", "Y", "YES")
                citations.append(
                    Citation(
                        patent_number=pn,
                        is_examiner=bool(is_examiner),
                        is_applicant="applicant" in category,
                    )
                )
        return citations

    def get_file_wrapper_text(
        self,
        app_number: str,
        codes: Sequence[str] = DEFAULT_FILE_WRAPPER_CODES,
        pdf_text_extractor: Optional[PdfTextExtractor] = None,
    ) -> str:
        """Download office actions/remarks PDFs and return their text.

        Documents are concatenated oldest-first, each prefixed by a
        ``=== CODE date ===`` header. Text extraction uses (in order) the
        ``pdf_text_extractor`` argument, the extractor passed at construction,
        or the PyMuPDF default.
        """
        extractor = (
            pdf_text_extractor or self._pdf_text_extractor or default_pdf_text_extractor
        )
        parts: list[str] = []
        for doc in self.get_documents_by_codes(app_number, codes):
            for url in doc.pdf_urls[:1]:  # one PDF rendition per document
                try:
                    text = extractor(self.download_pdf(url))
                except ImportError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Failed to extract %s document for application %s: %s",
                        doc.code, app_number, exc,
                    )
                    continue
                header = f"=== {doc.code} {doc.date or ''}".strip() + " ==="
                parts.append(f"{header}\n{text.strip()}")
        return "\n\n".join(parts)

    def enrich_patent(
        self,
        patent: Patent,
        *,
        codes: Sequence[str] = DEFAULT_FILE_WRAPPER_CODES,
        pdf_text_extractor: Optional[PdfTextExtractor] = None,
    ) -> Patent:
        """Return a copy of ``patent`` enriched with ODP file-wrapper data.

        Fills ``application_number``, ``file_wrapper_text``, examiner-cited
        citations (where ODP exposes them), and appends a
        ``SourceRecord(source="uspto_odp", fidelity=2)``.
        """
        app_number = patent.application_number or self.app_number_for_patent(
            patent.patent_number
        )
        if not app_number:
            raise ValueError(
                f"Could not resolve an application number for {patent.patent_number}"
            )
        enriched = patent.model_copy(deep=True)
        enriched.application_number = app_number

        try:
            new_citations = self.get_examiner_cited_art(app_number)
        except Exception as exc:  # citation bags are best-effort
            logger.warning("Citation lookup failed for %s: %s", app_number, exc)
            new_citations = []
        known = {str(c.patent_number) for c in enriched.citations}
        for citation in new_citations:
            if str(citation.patent_number) not in known:
                enriched.citations.append(citation)
                known.add(str(citation.patent_number))

        text = self.get_file_wrapper_text(
            app_number, codes=codes, pdf_text_extractor=pdf_text_extractor
        )
        if text:
            enriched.file_wrapper_text = text

        enriched.sources.append(
            SourceRecord(
                source="uspto_odp",
                fidelity=2,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                url=f"{ODP_BASE_URL}/{app_number}",
            )
        )
        return enriched
