"""Google Patents connectors.

Two clients:

- :class:`GooglePatentsScraper` — fetches and parses the public
  https://patents.google.com/patent/{number}/en page (itemprop microdata).
  No API key required; HTML parsing needs BeautifulSoup
  (``pip install patentkit[scrape]``). Highest-fidelity free source
  (fidelity=3): full text, claims, citations with examiner flags, CPC codes.
- :class:`SerpApiGooglePatentsSearch` — keyword/field search over Google
  Patents via SerpApi (https://serpapi.com/google-patents-api). Requires a
  SerpApi key (``SERPAPI_API_KEY``, https://serpapi.com/manage-api-key);
  returns lightweight Patent records (fidelity=1).

``fetch_patent(number)`` is a module-level convenience for the scraper.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional, Union

from patentkit.config import resolve_key
from patentkit.connectors.http import RateLimiter, request_json, request_text
from patentkit.models.patent import (
    Assignee,
    Citation,
    Claim,
    Classification,
    Inventor,
    Patent,
    PatentNumber,
    SourceRecord,
    SpecSection,
)

logger = logging.getLogger(__name__)

GOOGLE_PATENTS_URL = "https://patents.google.com/patent/{number}/en"
SERPAPI_URL = "https://serpapi.com/search"

#: Google Patents serves a different (harder to parse) page to obvious bots,
#: so the scraper identifies as a browser while keeping the patentkit tag.
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 patentkit/0.1"
)

_DEPENDENCY_RE = re.compile(r"\bclaims?\s+(\d+)", re.IGNORECASE)


def _require_bs4():
    try:
        from bs4 import BeautifulSoup  # noqa: F401

        return BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "Google Patents HTML parsing requires BeautifulSoup. Install it "
            "with `pip install patentkit[scrape]` (or `pip install beautifulsoup4`)."
        ) from exc


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _coerce_number(number: Union[str, PatentNumber]) -> PatentNumber:
    return number if isinstance(number, PatentNumber) else PatentNumber.parse(number)


class GooglePatentsScraper:
    """Scrape one patent's Google Patents page into a canonical Patent."""

    def __init__(
        self,
        *,
        min_interval_s: float = 0.0,
        timeout: float = 30.0,
        user_agent: str = SCRAPER_USER_AGENT,
    ):
        self._rate_limiter = RateLimiter(min_interval_s)
        self.timeout = timeout
        self.user_agent = user_agent

    def fetch(self, number: Union[str, PatentNumber]) -> Patent:
        """Fetch and parse a patent page; raises on HTTP errors / 404."""
        pn = _coerce_number(number)
        url = GOOGLE_PATENTS_URL.format(number=str(pn))
        html = request_text(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=self.timeout,
            rate_limiter=self._rate_limiter,
        )
        return self.parse_html(pn, html, url=url)

    ## Parsing (no network) ###############################################

    def parse_html(
        self,
        number: Union[str, PatentNumber],
        html: str,
        url: Optional[str] = None,
    ) -> Patent:
        """Parse a Google Patents HTML page into a canonical Patent."""
        BeautifulSoup = _require_bs4()
        pn = _coerce_number(number)
        soup = BeautifulSoup(html, "html.parser")

        specification, spec_sections = self._parse_description(soup)
        grant_date, expiration_date = self._parse_event_dates(soup)

        return Patent(
            patent_number=pn,
            title=self._itemprop_text(soup, "span", "title"),
            abstract=self._parse_abstract(soup),
            specification=specification,
            spec_sections=spec_sections,
            claims=self._parse_claims(soup),
            citations=self._parse_citations(soup, "backwardReferences")
            + self._parse_citations(soup, "backwardReferencesFamily", family=True),
            cited_by=self._parse_citations(soup, "forwardReferences")
            + self._parse_citations(soup, "forwardReferencesFamily", family=True),
            inventors=[
                Inventor(name=dd.get_text().strip())
                for dd in soup.find_all("dd", attrs={"itemprop": "inventor"})
                if dd.get_text().strip()
            ],
            assignees=self._parse_assignees(soup),
            classifications=self._parse_classifications(soup),
            priority_date=_parse_iso_date(self._time_text(soup, "priorityDate")),
            filing_date=_parse_iso_date(self._time_text(soup, "filingDate")),
            publication_date=_parse_iso_date(self._time_text(soup, "publicationDate")),
            grant_date=grant_date,
            expiration_date=expiration_date,
            status=self._itemprop_text(soup, "dd", "legalStatusIfi"),
            application_number=self._parse_application_number(soup),
            sources=[
                SourceRecord(
                    source="google_patents",
                    fidelity=3,
                    url=url or GOOGLE_PATENTS_URL.format(number=str(pn)),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )
            ],
        )

    @staticmethod
    def _itemprop_text(soup: Any, tag: str, itemprop: str) -> Optional[str]:
        el = soup.find(tag, attrs={"itemprop": itemprop})
        if el:
            text = el.get_text().strip()
            return text or None
        return None

    @staticmethod
    def _time_text(soup: Any, itemprop: str) -> Optional[str]:
        el = soup.find("time", attrs={"itemprop": itemprop})
        return el.get_text().strip() if el else None

    @staticmethod
    def _parse_abstract(soup: Any) -> Optional[str]:
        el = soup.find("div", class_="abstract") or soup.find(
            "section", attrs={"itemprop": "abstract"}
        )
        if el:
            text = el.get_text().strip()
            return text or None
        return None

    @staticmethod
    def _parse_description(soup: Any) -> tuple[Optional[str], list[SpecSection]]:
        section = soup.find("section", attrs={"itemprop": "description"})
        if section is None:
            return None, []
        from bs4 import NavigableString

        specification = re.sub(r"\n{3,}", "\n\n", section.get_text("\n")).strip()

        sections: list[SpecSection] = []
        heading: Optional[str] = None
        buffer: list[str] = []

        def flush() -> None:
            text = " ".join(part for part in buffer if part).strip()
            if text:
                sections.append(SpecSection(heading=heading, text=text))

        for node in section.descendants:
            name = getattr(node, "name", None)
            if name == "heading" or (name == "h2" and node is not section):
                flush()
                heading = node.get_text().strip()
                buffer = []
            elif isinstance(node, NavigableString):
                parent_name = getattr(node.parent, "name", None)
                if parent_name in ("heading", "h2"):
                    continue
                text = str(node).strip()
                if text:
                    buffer.append(text)
        flush()
        return specification or None, sections

    @staticmethod
    def _parse_claims(soup: Any) -> list[Claim]:
        claims_section = soup.find("section", attrs={"itemprop": "claims"})
        if claims_section is None:
            return []
        claims: list[Claim] = []
        seen: set[int] = set()
        for div in claims_section.find_all(
            "div", class_="claim", attrs={"num": re.compile(r"\d+")}
        ):
            try:
                num = int(div.attrs["num"].split("-")[0].lstrip("0") or "0")
            except ValueError:
                continue
            if num in seen:
                continue
            seen.add(num)
            text = re.sub(r"\s+", " ", div.get_text(" ")).strip()
            depends_on: Optional[int] = None
            ref = div.find("claim-ref")
            ref_text = ref.get_text() if ref else text
            match = _DEPENDENCY_RE.search(ref_text)
            if match:
                candidate = int(match.group(1))
                if candidate != num:
                    depends_on = candidate
            claims.append(Claim(number=num, text=text, depends_on=depends_on))
        claims.sort(key=lambda c: c.number)
        return claims

    @staticmethod
    def _parse_citations(soup: Any, itemprop: str, family: bool = False) -> list[Citation]:
        citations: list[Citation] = []
        for row in soup.find_all("tr", attrs={"itemprop": itemprop}):
            number_el = row.find("span", attrs={"itemprop": "publicationNumber"})
            if not number_el:
                continue
            raw = number_el.get_text().strip()
            # Examiner-cited art is flagged with an examinerCited span (the
            # rendered page shows it as an asterisk on the number).
            is_examiner = bool(
                row.find("span", attrs={"itemprop": "examinerCited"})
            ) or raw.endswith("*")
            raw = raw.rstrip("*").strip()
            try:
                pn = PatentNumber.parse(raw)
            except ValueError:
                continue
            citations.append(
                Citation(
                    patent_number=pn,
                    is_examiner=is_examiner,
                    is_third_party=bool(
                        row.find("span", attrs={"itemprop": "thirdPartyCited"})
                    ),
                    is_family_to_family=family,
                )
            )
        return citations

    @staticmethod
    def _parse_assignees(soup: Any) -> list[Assignee]:
        assignees = [
            Assignee(name=dd.get_text().strip())
            for dd in soup.find_all("dd", attrs={"itemprop": "assigneeCurrent"})
            if dd.get_text().strip()
        ]
        if not assignees:
            assignees = [
                Assignee(name=dd.get_text().strip())
                for dd in soup.find_all("dd", attrs={"itemprop": "assigneeOriginal"})
                if dd.get_text().strip()
            ]
        return assignees

    @staticmethod
    def _parse_classifications(soup: Any) -> list[Classification]:
        leaf: list[Classification] = []
        all_codes: list[Classification] = []
        seen: set[str] = set()
        for ul in soup.find_all("ul", attrs={"itemprop": "classifications"}):
            for li in ul.find_all("li"):
                code_el = li.find("span", attrs={"itemprop": "Code"})
                if not code_el:
                    continue
                code = code_el.get_text().strip()
                if not code or code in seen:
                    continue
                seen.add(code)
                desc_el = li.find("span", attrs={"itemprop": "Description"})
                classification = Classification(
                    scheme="CPC",
                    code=code,
                    description=desc_el.get_text().strip() if desc_el else None,
                )
                all_codes.append(classification)
                leaf_meta = li.find("meta", attrs={"itemprop": "Leaf"})
                if leaf_meta is not None and leaf_meta.attrs.get("content") == "true":
                    leaf.append(classification)
        return leaf or all_codes

    @staticmethod
    def _parse_event_dates(soup: Any) -> tuple[Optional[date], Optional[date]]:
        grant_date: Optional[date] = None
        expiration_date: Optional[date] = None
        for event in soup.find_all("dd", attrs={"itemprop": "events"}):
            title_el = event.find("span", attrs={"itemprop": "title"})
            type_el = event.find("span", attrs={"itemprop": "type"})
            time_el = event.find("time", attrs={"itemprop": "date"})
            if time_el is None:
                continue
            when = _parse_iso_date(time_el.attrs.get("datetime") or time_el.get_text())
            title = title_el.get_text().strip().lower() if title_el else ""
            event_type = type_el.get_text().strip().lower() if type_el else ""
            if when and (event_type == "granted" or "application granted" in title):
                grant_date = when
            if when and "expiration" in title:
                expiration_date = when
        return grant_date, expiration_date

    @staticmethod
    def _parse_application_number(soup: Any) -> Optional[str]:
        el = soup.find("dd", attrs={"itemprop": "applicationNumber"})
        if el and el.get_text().strip():
            return el.get_text().strip()
        link = soup.find(
            "a", href=lambda href: bool(href and "patentcenter.uspto.gov" in href)
        )
        if link:
            return link["href"].rstrip("/").split("/")[-1]
        return None


class SerpApiGooglePatentsSearch:
    """Google Patents keyword search via SerpApi (lightweight records)."""

    def __init__(self, api_key: Optional[str] = None, *, min_interval_s: float = 0.0):
        self.api_key = resolve_key("SERPAPI_API_KEY", api_key)
        self._rate_limiter = RateLimiter(min_interval_s)

    def search(
        self,
        q: str,
        *,
        inventor: Optional[str] = None,
        assignee: Optional[str] = None,
        before: Optional[Union[str, date]] = None,
        after: Optional[Union[str, date]] = None,
        country: Optional[str] = None,
        status: Optional[str] = None,
        num: int = 20,
        page: Optional[int] = None,
    ) -> list[Patent]:
        """Search Google Patents; returns lightweight Patents (fidelity=1).

        ``before``/``after`` accept either a serpapi-style string (e.g.
        ``"priority:20221231"``) or a :class:`datetime.date` (interpreted as a
        priority-date bound). ``status`` is ``"GRANT"`` or ``"APPLICATION"``.
        """
        params: dict[str, Any] = {
            "engine": "google_patents",
            "q": q,
            "num": num,
            "api_key": self.api_key,
        }
        if inventor:
            params["inventor"] = inventor
        if assignee:
            params["assignee"] = assignee
        if before:
            params["before"] = (
                before if isinstance(before, str) else f"priority:{before:%Y%m%d}"
            )
        if after:
            params["after"] = (
                after if isinstance(after, str) else f"priority:{after:%Y%m%d}"
            )
        if country:
            params["country"] = country
        if status:
            params["status"] = status
        if page is not None:
            params["page"] = page

        data = request_json(
            "GET", SERPAPI_URL, params=params, rate_limiter=self._rate_limiter
        )
        return [
            patent
            for result in data.get("organic_results", []) or []
            if (patent := self._result_to_patent(result)) is not None
        ]

    @staticmethod
    def _result_to_patent(result: dict[str, Any]) -> Optional[Patent]:
        raw_number = result.get("publication_number") or result.get("patent_id") or ""
        raw_number = raw_number.removeprefix("patent/").removesuffix("/en")
        try:
            pn = PatentNumber.parse(raw_number)
        except ValueError:
            logger.debug("Skipping unparseable serpapi result: %r", raw_number)
            return None
        assignee = result.get("assignee")
        inventor = result.get("inventor")
        return Patent(
            patent_number=pn,
            title=(result.get("title") or "").strip() or None,
            abstract=result.get("snippet") or None,
            assignees=[Assignee(name=assignee)] if assignee else [],
            inventors=[Inventor(name=inventor)] if inventor else [],
            priority_date=_parse_iso_date(result.get("priority_date")),
            filing_date=_parse_iso_date(result.get("filing_date")),
            publication_date=_parse_iso_date(result.get("publication_date")),
            grant_date=_parse_iso_date(result.get("grant_date")),
            sources=[
                SourceRecord(
                    source="serpapi_google_patents",
                    fidelity=1,
                    url=result.get("patent_link") or result.get("link"),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )
            ],
        )


def fetch_patent(number: Union[str, PatentNumber]) -> Patent:
    """Convenience: scrape one patent from Google Patents."""
    return GooglePatentsScraper().fetch(number)
