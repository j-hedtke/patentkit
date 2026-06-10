"""EPO Open Patent Services (OPS) client.

Data source: https://ops.epo.org/3.2 — worldwide bibliographic data and
INPADOC patent families from the European Patent Office.

Auth: OAuth2 client-credentials. Register an app at
https://developers.epo.org/ to get a consumer key + secret, then set
``EPO_OPS_KEY`` and ``EPO_OPS_SECRET`` (or pass ``key=``/``secret=``).
Access tokens are fetched from ``/3.2/auth/accesstoken`` and cached until
shortly before expiry.

OPS is the lowest-fidelity connector (fidelity=1): great family coverage,
bibliographic data only (no full text here).
"""

from __future__ import annotations

import logging
import time
from base64 import b64encode
from datetime import date, datetime, timezone
from typing import Optional, Union
from xml.etree import ElementTree as ET

import httpx

from patentkit.config import resolve_key
from patentkit.connectors.http import (
    DEFAULT_USER_AGENT,
    RateLimiter,
    request_json,
)
from patentkit.models.patent import (
    Assignee,
    Classification,
    Inventor,
    Patent,
    PatentNumber,
    SourceRecord,
)

logger = logging.getLogger(__name__)

OPS_AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
OPS_BASE_URL = "https://ops.epo.org/3.2/rest-services"

_NS = {
    "ops": "http://ops.epo.org",
    "ex": "http://www.epo.org/exchange",
}


def _parse_yyyymmdd(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    value = value.strip()
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except (ValueError, IndexError):
        return None


def _coerce_number(number: Union[str, PatentNumber]) -> PatentNumber:
    return number if isinstance(number, PatentNumber) else PatentNumber.parse(number)


def _doc_id_path(pn: PatentNumber) -> str:
    """OPS reference path: docdb format when the kind code is known."""
    if pn.kind_code:
        return f"docdb/{pn.country_code}.{pn.number}.{pn.kind_code}"
    return f"epodoc/{pn.country_code}{pn.number}"


class EpoOpsClient:
    """Minimal OPS client: INPADOC family + bibliographic data."""

    def __init__(
        self,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        *,
        min_interval_s: float = 0.0,
        timeout: float = 30.0,
    ):
        self.key = resolve_key("EPO_OPS_KEY", key)
        self.secret = resolve_key("EPO_OPS_SECRET", secret)
        self.timeout = timeout
        self._rate_limiter = RateLimiter(min_interval_s)
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    ## Auth ###############################################################

    def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        basic = b64encode(f"{self.key}:{self.secret}".encode("ascii")).decode("ascii")
        data = request_json(
            "POST",
            OPS_AUTH_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=self.timeout,
        )
        self._token = data["access_token"]
        # refresh a minute early
        self._token_expires_at = time.monotonic() + int(data.get("expires_in", 1200)) - 60
        return self._token

    def _get_xml(self, path: str) -> ET.Element:
        self._rate_limiter.wait()
        response = httpx.get(
            f"{OPS_BASE_URL}/{path}",
            headers={
                "Authorization": f"Bearer {self._access_token()}",
                "Accept": "application/xml",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            timeout=self.timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        return ET.fromstring(response.content)

    ## Family #############################################################

    def get_family(self, patent_number: Union[str, PatentNumber]) -> list[PatentNumber]:
        """INPADOC (simple) family members of a publication."""
        pn = _coerce_number(patent_number)
        root = self._get_xml(f"family/publication/{_doc_id_path(pn)}")
        members: list[PatentNumber] = []
        seen: set[str] = set()
        for member in root.findall(".//ops:family-member", _NS):
            doc_id = member.find(
                ".//ex:publication-reference/ex:document-id[@document-id-type='docdb']",
                _NS,
            )
            if doc_id is None:
                continue
            country = doc_id.findtext("ex:country", default="", namespaces=_NS).strip()
            number = doc_id.findtext("ex:doc-number", default="", namespaces=_NS).strip()
            kind = doc_id.findtext("ex:kind", default="", namespaces=_NS).strip()
            if not number:
                continue
            family_pn = PatentNumber(
                country_code=country or "US", number=number, kind_code=kind or None
            )
            if str(family_pn) not in seen:
                seen.add(str(family_pn))
                members.append(family_pn)
        return members

    ## Bibliographic data #################################################

    def get_biblio(self, patent_number: Union[str, PatentNumber]) -> Patent:
        """Bibliographic record (title, abstract, parties, dates, IPC/CPC)."""
        pn = _coerce_number(patent_number)
        path = f"published-data/publication/{_doc_id_path(pn)}/biblio"
        root = self._get_xml(path)
        doc = root.find(".//ex:exchange-document", _NS)
        if doc is None:
            raise ValueError(f"OPS returned no exchange-document for {pn}")

        title = self._best_lang_text(doc.findall(".//ex:invention-title", _NS))
        abstract = self._best_lang_text(doc.findall(".//ex:abstract", _NS))

        pub_doc_id = doc.find(
            ".//ex:publication-reference/ex:document-id[@document-id-type='docdb']", _NS
        )
        publication_date = (
            _parse_yyyymmdd(pub_doc_id.findtext("ex:date", namespaces=_NS))
            if pub_doc_id is not None
            else None
        )
        app_doc_id = doc.find(".//ex:application-reference/ex:document-id", _NS)
        filing_date = (
            _parse_yyyymmdd(app_doc_id.findtext("ex:date", namespaces=_NS))
            if app_doc_id is not None
            else None
        )
        priority_dates = [
            d
            for el in doc.findall(".//ex:priority-claim//ex:date", _NS)
            if (d := _parse_yyyymmdd(el.text)) is not None
        ]

        inventors = self._parse_party(doc, ".//ex:inventors/ex:inventor", "ex:inventor-name")
        applicants = self._parse_party(doc, ".//ex:applicants/ex:applicant", "ex:applicant-name")

        return Patent(
            patent_number=pn,
            title=title,
            abstract=abstract,
            inventors=[Inventor(name=n) for n in inventors],
            assignees=[Assignee(name=n) for n in applicants],
            classifications=self._parse_classifications(doc),
            priority_date=min(priority_dates) if priority_dates else None,
            filing_date=filing_date,
            publication_date=publication_date,
            sources=[
                SourceRecord(
                    source="epo_ops",
                    fidelity=1,
                    url=f"{OPS_BASE_URL}/{path}",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )
            ],
        )

    @staticmethod
    def _best_lang_text(elements: list[ET.Element]) -> Optional[str]:
        """English text where available, otherwise the first entry."""
        chosen: Optional[ET.Element] = None
        for el in elements:
            if el.get("lang", "").lower() == "en":
                chosen = el
                break
        if chosen is None and elements:
            chosen = elements[0]
        if chosen is None:
            return None
        text = " ".join("".join(chosen.itertext()).split()).strip()
        return text or None

    @staticmethod
    def _parse_party(doc: ET.Element, party_path: str, name_tag: str) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        # prefer "original" data-format entries; OPS duplicates each party in
        # epodoc format (UPPER CASE [country]) too.
        parties = doc.findall(party_path, _NS)
        originals = [p for p in parties if p.get("data-format") == "original"]
        for party in originals or parties:
            name = party.findtext(f"{name_tag}/ex:name", default="", namespaces=_NS)
            name = " ".join(name.split()).strip().rstrip(",")
            if name and name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
        return names

    @staticmethod
    def _parse_classifications(doc: ET.Element) -> list[Classification]:
        classifications: list[Classification] = []
        seen: set[tuple[str, str]] = set()
        for el in doc.findall(".//ex:classifications-ipcr/ex:classification-ipcr/ex:text", _NS):
            code = " ".join((el.text or "").split())
            if code and ("IPC", code) not in seen:
                seen.add(("IPC", code))
                classifications.append(Classification(scheme="IPC", code=code))
        for pc in doc.findall(".//ex:patent-classifications/ex:patent-classification", _NS):
            scheme_el = pc.find("ex:classification-scheme", _NS)
            scheme = (scheme_el.get("scheme", "") if scheme_el is not None else "").upper()
            scheme = "CPC" if scheme.startswith("CPC") else (scheme or "CPC")
            section = pc.findtext("ex:section", default="", namespaces=_NS).strip()
            if not section:
                continue
            code = (
                section
                + pc.findtext("ex:class", default="", namespaces=_NS).strip()
                + pc.findtext("ex:subclass", default="", namespaces=_NS).strip()
                + pc.findtext("ex:main-group", default="", namespaces=_NS).strip()
            )
            subgroup = pc.findtext("ex:subgroup", default="", namespaces=_NS).strip()
            if subgroup:
                code += f"/{subgroup}"
            if (scheme, code) not in seen:
                seen.add((scheme, code))
                classifications.append(Classification(scheme=scheme, code=code))
        return classifications
