"""USPTO bulk full-text ("redbook") ingestion.

Data source: https://bulkdata.uspto.gov/data/patent/{grant|application}/redbook/fulltext/{year}/
— free weekly zip archives of every granted patent (``ipgYYMMDD.zip``,
published Tuesdays) or published application (``ipaYYMMDD.zip``, Thursdays).
Each archive holds one giant file of concatenated XML documents, split here
on ``<?xml`` declarations and parsed into canonical
:class:`~patentkit.models.patent.Patent` records (fidelity=2). No API key
required.

Design patents (``D...``) and reissues (``RE...``) are skipped.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterator, Optional, Union
from xml.etree import ElementTree as ET

from patentkit.connectors.http import download
from patentkit.models.patent import (
    Assignee,
    Claim,
    Classification,
    Inventor,
    Patent,
    PatentNumber,
    SourceRecord,
)

logger = logging.getLogger(__name__)

BULK_BASE_URL = "https://bulkdata.uspto.gov/data/patent"

_DOCTYPE_RE = re.compile(r"<!DOCTYPE.*?\]?>", re.DOTALL)
_CLAIM_DEP_RE = re.compile(r"\bclaims?\s+(\d+)", re.IGNORECASE)
_XML_DECL = b'<?xml version="1.0"'


def weekly_archive_urls(year: int, kind: str = "grant") -> list[str]:
    """Urls of every weekly redbook archive for ``year``.

    Grants are published Tuesdays (``ipg``), applications Thursdays
    (``ipa``). Occasional holiday weeks may 404 — skip those.
    """
    if kind not in ("grant", "application"):
        raise ValueError(f"kind must be 'grant' or 'application', got {kind!r}")
    letter = "g" if kind == "grant" else "a"
    weekday = 1 if kind == "grant" else 3  # Mon=0
    day = dt.date(year, 1, 1)
    while day.weekday() != weekday:
        day += dt.timedelta(days=1)
    urls: list[str] = []
    while day.year == year:
        urls.append(
            f"{BULK_BASE_URL}/{kind}/redbook/fulltext/{year}/ip{letter}{day:%y%m%d}.zip"
        )
        day += dt.timedelta(weeks=1)
    return urls


def download_archive(url: str, dest: Union[str, Path]) -> Path:
    """Download one weekly archive to ``dest``."""
    path = download(url, dest)
    assert isinstance(path, Path)
    return path


def iter_patents_from_archive(zip_path: Union[str, Path]) -> Iterator[Patent]:
    """Yield canonical Patents from a downloaded redbook zip.

    Documents that fail to parse (or are design/reissue patents) are
    logged and skipped.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".xml"):
                continue
            raw = archive.read(name)
            for chunk in raw.split(_XML_DECL):
                if not chunk.strip():
                    continue
                xml_text = (_XML_DECL + chunk).decode("utf-8", errors="replace")
                patent = parse_redbook_xml(xml_text, raw_ref=f"{zip_path.name}:{name}")
                if patent is not None:
                    yield patent


def _parse_yyyymmdd(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    value = value.strip()
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except (ValueError, IndexError):
        return None


def _element_text(element: Optional[ET.Element]) -> Optional[str]:
    if element is None:
        return None
    text = re.sub(r"\n{3,}", "\n\n", "".join(element.itertext())).strip()
    return text or None


def _parse_claims(root: ET.Element) -> list[Claim]:
    claims: list[Claim] = []
    for claim_el in root.iter("claim"):
        num_attr = (claim_el.get("num") or "").split("-")[0]
        digits = num_attr.lstrip("0")
        if not digits.isdigit():
            continue
        number = int(digits)
        text = re.sub(r"\s+", " ", "".join(claim_el.itertext())).strip()
        depends_on: Optional[int] = None
        ref = claim_el.find(".//claim-ref")
        if ref is not None:
            idref = ref.get("idref") or ""
            ref_digits = re.sub(r"\D", "", idref)
            if ref_digits:
                depends_on = int(ref_digits)
            else:
                match = _CLAIM_DEP_RE.search("".join(ref.itertext()))
                if match:
                    depends_on = int(match.group(1))
        if depends_on == number:
            depends_on = None
        claims.append(Claim(number=number, text=text, depends_on=depends_on))
    return claims


def _parse_cpc(root: ET.Element) -> list[Classification]:
    classifications: list[Classification] = []
    seen: set[str] = set()
    for cpc in root.iter("classification-cpc"):
        section = cpc.findtext("section", "").strip()
        if not section:
            continue
        code = (
            section
            + cpc.findtext("class", "").strip()
            + cpc.findtext("subclass", "").strip()
            + cpc.findtext("main-group", "").strip()
        )
        subgroup = cpc.findtext("subgroup", "").strip()
        if subgroup:
            code += f"/{subgroup}"
        if code not in seen:
            seen.add(code)
            classifications.append(Classification(scheme="CPC", code=code))
    return classifications


def parse_redbook_xml(xml_text: str, *, raw_ref: Optional[str] = None) -> Optional[Patent]:
    """Parse one redbook XML document into a canonical Patent.

    Returns None for unparseable documents and for design (``D``) /
    reissue (``RE``) patents.
    """
    xml_text = _DOCTYPE_RE.sub("", xml_text)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("Skipping unparseable redbook document (%s): %s", raw_ref, exc)
        return None

    pub = root.find(".//publication-reference/document-id")
    if pub is None:
        return None
    doc_number = (pub.findtext("doc-number") or "").strip()
    if not doc_number:
        return None
    if doc_number.startswith("D") or doc_number.startswith("RE"):
        logger.debug("Skipping design/reissue patent %s", doc_number)
        return None

    patent_number = PatentNumber(
        country_code=(pub.findtext("country") or "US").strip(),
        number=doc_number.lstrip("0"),
        kind_code=(pub.findtext("kind") or "").strip() or None,
    )
    publication_date = _parse_yyyymmdd(pub.findtext("date"))

    app = root.find(".//application-reference/document-id")
    filing_date = _parse_yyyymmdd(app.findtext("date")) if app is not None else None
    application_number = (
        (app.findtext("doc-number") or "").strip() or None if app is not None else None
    )

    priority_dates = [
        d
        for claim in root.iter("priority-claim")
        if (d := _parse_yyyymmdd(claim.findtext("date"))) is not None
    ]
    priority_date = min(priority_dates) if priority_dates else None

    inventors: list[Inventor] = []
    for inventor in root.iter("inventor"):
        last = (inventor.findtext(".//last-name") or "").strip()
        first = (inventor.findtext(".//first-name") or "").strip()
        if last and first:
            inventors.append(Inventor(name=f"{last}, {first}"))
        elif last or first:
            inventors.append(Inventor(name=last or first))

    assignees = [
        Assignee(name=name.strip())
        for assignee in root.iter("assignee")
        if (name := assignee.findtext(".//orgname"))
        and name.strip()
    ]

    is_grant = root.tag == "us-patent-grant" or root.find(
        ".//us-bibliographic-data-grant"
    ) is not None

    return Patent(
        patent_number=patent_number,
        title=_element_text(root.find(".//invention-title")),
        abstract=_element_text(root.find(".//abstract")),
        specification=_element_text(root.find(".//description")),
        claims=_parse_claims(root),
        inventors=inventors,
        assignees=assignees,
        classifications=_parse_cpc(root),
        priority_date=priority_date,
        filing_date=filing_date,
        publication_date=publication_date,
        grant_date=publication_date if is_grant else None,
        application_number=application_number,
        sources=[
            SourceRecord(source="uspto_bulk", fidelity=2, raw_ref=raw_ref)
        ],
    )
