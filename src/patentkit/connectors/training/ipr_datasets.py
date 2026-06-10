"""IPR outcome datasets (evaluation data for invalidity-search models).

Final written decisions in inter partes reviews record which prior-art
references were actually used against which claims — ground truth for
evaluating prior-art search. This module turns PTAB final-decision IPRs
into JSONL :class:`IprEvalRecord` rows: proceeding, challenged patent,
prior-art references mentioned in the final decision, and outcome.

Reference extraction from decision text is injectable; the default is a
regex over US patent number forms ("U.S. Patent No. 7,654,321",
"US 8123456", ...). PDF text extraction defaults to PyMuPDF
(``pip install patentkit[pdf]``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, Field

from patentkit.connectors.infra.ptab import (
    FINAL_STATUS_CATEGORIES,
    IprProceeding,
    PtabClient,
)

logger = logging.getLogger(__name__)

#: callable mapping final-decision text -> prior-art reference identifiers
ReferenceExtractor = Callable[[str], list[str]]


class IprEvalRecord(BaseModel):
    """One IPR final decision, reduced to an evaluation example."""

    proceeding_number: str
    challenged_patent: str
    claims: list[int] = Field(default_factory=list)
    prior_art_references: list[str] = Field(default_factory=list)
    outcome: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


#: "U.S. Patent No. 7,654,321" / "7,654,321" / "US 8123456" / "US8,123,456"
_US_PATENT_RE = re.compile(
    r"\b(?:U\.?S\.?\s*(?:Pat(?:ent)?\.?\s*(?:No\.?)?)?\s*)?(\d{1,2},\d{3},\d{3})\b"
    r"|\bUS[-\s]?(\d{7,8})\b",
    re.IGNORECASE,
)


def extract_us_patent_numbers(text: str) -> list[str]:
    """Default reference extractor: normalized US patent numbers, in order."""
    numbers: list[str] = []
    seen: set[str] = set()
    for match in _US_PATENT_RE.finditer(text):
        number = (match.group(1) or match.group(2)).replace(",", "")
        normalized = f"US{number}"
        if normalized not in seen:
            seen.add(normalized)
            numbers.append(normalized)
    return numbers


def _default_pdf_text(pdf_bytes: bytes) -> str:
    from patentkit.connectors.inference.file_wrapper import default_pdf_text_extractor

    return default_pdf_text_extractor(pdf_bytes)


def build_ipr_eval_dataset(
    ptab: PtabClient,
    filed_from: str,
    filed_to: str,
    out_path: Union[str, Path],
    limit: Optional[int] = None,
    reference_extractor: Optional[ReferenceExtractor] = None,
    pdf_text_extractor: Optional[Callable[[bytes], str]] = None,
) -> list[IprEvalRecord]:
    """Build a JSONL eval dataset from final-decision IPRs.

    Iterates IPR proceedings filed in ``[filed_from, filed_to]``
    (``YYYY-MM-DD``), keeps those with a final decision, downloads the final
    written decision, extracts prior-art references from its text, and
    appends one :class:`IprEvalRecord` JSON line per proceeding to
    ``out_path``. Returns the records written. Proceedings that fail are
    logged and skipped.
    """
    extract_refs = reference_extractor or extract_us_patent_numbers
    extract_text = pdf_text_extractor or _default_pdf_text
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[IprEvalRecord] = []
    with out_path.open("a", encoding="utf-8") as out:
        for proceeding in ptab.iter_ipr_proceedings(filed_from, filed_to):
            if limit is not None and len(records) >= limit:
                break
            if proceeding.status not in FINAL_STATUS_CATEGORIES:
                continue
            try:
                record = _proceeding_to_record(
                    ptab, proceeding, extract_refs, extract_text
                )
            except Exception as exc:
                logger.warning(
                    "Skipping proceeding %s: %s", proceeding.proceeding_number, exc
                )
                continue
            if record is None:
                continue
            out.write(record.model_dump_json() + "\n")
            out.flush()
            records.append(record)
    return records


def _proceeding_to_record(
    ptab: PtabClient,
    proceeding: IprProceeding,
    extract_refs: ReferenceExtractor,
    extract_text: Callable[[bytes], str],
) -> Optional[IprEvalRecord]:
    if not proceeding.patent_number:
        return None
    documents = ptab.list_proceeding_documents(proceeding.proceeding_number)
    final_doc = next((d for d in documents if d.is_final_decision), None)
    if final_doc is None:
        logger.info(
            "No final decision document in %s", proceeding.proceeding_number
        )
        return None
    pdf_bytes = ptab.download_document(final_doc.document_id)
    assert isinstance(pdf_bytes, bytes)
    text = extract_text(pdf_bytes)
    references = extract_refs(text)
    challenged = f"US{proceeding.patent_number}".replace("USUS", "US")
    # the challenged patent itself is always cited in the decision
    references = [r for r in references if r != challenged]
    return IprEvalRecord(
        proceeding_number=proceeding.proceeding_number,
        challenged_patent=challenged,
        prior_art_references=references,
        outcome=proceeding.status,
        metadata={
            "filing_date": proceeding.filing_date,
            "petitioner": proceeding.petitioner,
            "patent_owner": proceeding.patent_owner,
            "final_document_id": final_doc.document_id,
            "final_document_title": final_doc.title,
        },
    )


def load_ipr_eval_dataset(path: Union[str, Path]) -> list[IprEvalRecord]:
    """Load a JSONL dataset written by :func:`build_ipr_eval_dataset`."""
    records: list[IprEvalRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(IprEvalRecord(**json.loads(line)))
    return records
