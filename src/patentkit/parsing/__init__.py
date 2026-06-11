"""Parsers: claim text -> canonical model, patent PDF line-number location,
and generic document text extraction.

Everything here is importable with zero optional dependencies; functions that
need ``pymupdf``/``rapidfuzz``/``python-docx``/``bs4`` import them lazily and
raise ImportError naming the pip extra.
"""

from patentkit.parsing.claims import (
    claim_element_outline,
    extract_claims_section,
    parse_claims,
    split_limitations,
)
from patentkit.parsing.documents import extract_text, html_to_text
from patentkit.parsing.patent_pdf import (
    LineMarker,
    LineNumberModel,
    PassageLocation,
    PdfTextLine,
    extract_patent_page_lines,
    filter_marker_candidates,
    fit_line_model,
    format_patent_citation,
    locate_passage,
)

__all__ = [
    "claim_element_outline",
    "extract_claims_section",
    "parse_claims",
    "split_limitations",
    "extract_text",
    "html_to_text",
    "LineMarker",
    "LineNumberModel",
    "PassageLocation",
    "PdfTextLine",
    "extract_patent_page_lines",
    "filter_marker_candidates",
    "fit_line_model",
    "format_patent_citation",
    "locate_passage",
]
