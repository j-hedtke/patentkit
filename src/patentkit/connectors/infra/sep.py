"""Standard-essential patent (SEP) declaration materials.

The primary public source is the ETSI IPR database (https://ipr.etsi.org),
which records declarations that a patent is (claimed to be) essential to an
ETSI standard (3G/4G/5G, DECT, etc.). ETSI does not offer a public REST
API — declarations are obtained as CSV/Excel exports from the portal's
search screens. :func:`load_etsi_declarations_csv` parses such an export,
matching columns liberally because the export format has changed over time.

Use :func:`sep_patents_for_standard` to slice declarations down to a
standard family (e.g. everything declared against ``"EN 301 908"``).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SepDeclaration(BaseModel):
    """One row of an SEP declaration database export."""

    declaring_company: Optional[str] = None
    patent_number: Optional[str] = None
    standard: Optional[str] = None  # e.g. "ETSI EN 301 908"
    project: Optional[str] = None  # e.g. "5G/NR"
    declaration_date: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", header.lower()).strip()


#: canonical field -> normalized header fragments to match (first hit wins,
#: in listed order)
_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "declaring_company": [
        "declaring company",
        "company name",
        "declarant",
        "company",
    ],
    "patent_number": [
        "publication number",
        "patent number",
        "application number",
        "granted patent",
        "patent",
        "publication",
        "application",
    ],
    "standard": [
        "specification number",
        "standard number",
        "specification",
        "standard",
        "spec",
    ],
    "project": [
        "project",
        "technology",
        "committee",
    ],
    "declaration_date": [
        "declaration date",
        "signature date",
        "date",
    ],
}


def _map_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """Map canonical field names to actual CSV headers, liberally."""
    normalized = {name: _normalize_header(name) for name in fieldnames if name}
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for field_name, candidates in _COLUMN_CANDIDATES.items():
        for candidate in candidates:
            for original, norm in normalized.items():
                if original in used:
                    continue
                if candidate == norm or candidate in norm:
                    mapping[field_name] = original
                    used.add(original)
                    break
            if field_name in mapping:
                break
    return mapping


def load_etsi_declarations_csv(path: Union[str, Path]) -> list[SepDeclaration]:
    """Parse an ETSI IPR database CSV export into declarations.

    Column matching is intentionally liberal (e.g. "Declaring Company" /
    "Company", "Application Number" / "Patent Number" / "Publication Number",
    "Specification Number" / "Standard"). Unmapped columns are preserved in
    each declaration's ``raw`` dict.
    """
    path = Path(path)
    declarations: list[SepDeclaration] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        mapping = _map_columns(reader.fieldnames)
        if not mapping:
            logger.warning(
                "No recognizable columns in %s (headers: %s)", path, reader.fieldnames
            )
        for row in reader:
            values = {
                field_name: (row.get(header) or "").strip() or None
                for field_name, header in mapping.items()
            }
            declarations.append(SepDeclaration(**values, raw=dict(row)))
    return declarations


def sep_patents_for_standard(
    declarations: Iterable[SepDeclaration], standard_prefix: str
) -> list[str]:
    """Patent numbers declared essential to standards matching a prefix.

    Matching is case-insensitive and whitespace-insensitive, so
    ``"EN 301 908"`` matches ``"ETSI EN 301 908-1"`` etc. when the prefix
    appears at the start of (or anywhere in) the declared standard.
    """
    needle = re.sub(r"\s+", "", standard_prefix).lower()
    numbers: list[str] = []
    seen: set[str] = set()
    for declaration in declarations:
        if not declaration.patent_number or not declaration.standard:
            continue
        haystack = re.sub(r"\s+", "", declaration.standard).lower()
        if needle in haystack and declaration.patent_number not in seen:
            seen.add(declaration.patent_number)
            numbers.append(declaration.patent_number)
    return numbers


class EtsiSepConnector:
    """Documented stub for live ETSI IPR database access.

    ETSI's IPR database (https://ipr.etsi.org) has no public API; data is
    obtained interactively: search by company/standard/project on the portal
    and use its "Export" function to download a CSV. Feed that file to
    :func:`load_etsi_declarations_csv`.
    """

    PORTAL_URL = "https://ipr.etsi.org"

    def fetch(self, *args: Any, **kwargs: Any) -> list[SepDeclaration]:
        raise NotImplementedError(
            "ETSI's IPR database has no public API. Export declarations as CSV "
            f"from {self.PORTAL_URL} (search, then Export) and load them with "
            "patentkit.connectors.infra.sep.load_etsi_declarations_csv(path)."
        )
