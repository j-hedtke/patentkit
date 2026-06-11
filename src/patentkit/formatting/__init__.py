"""Document formatting: claim charts (markdown/HTML/DOCX) and search reports.

Markdown and HTML renderers are dependency-free; DOCX output imports
python-docx lazily (``pip install 'patentkit[docx]'``).
"""

from patentkit.formatting.claim_chart import (
    claim_chart_docx,
    claim_chart_html,
    claim_chart_markdown,
    filter_chart,
    limitation_chart_markdown,
)
from patentkit.formatting.reports import (
    fto_report_docx,
    fto_report_markdown,
    infringement_report_docx,
    infringement_report_markdown,
    invalidity_report_docx,
    invalidity_report_markdown,
)
from patentkit.formatting.trace import search_trace_markdown

__all__ = [
    "claim_chart_docx",
    "claim_chart_html",
    "claim_chart_markdown",
    "filter_chart",
    "limitation_chart_markdown",
    "search_trace_markdown",
    "fto_report_docx",
    "fto_report_markdown",
    "infringement_report_docx",
    "infringement_report_markdown",
    "invalidity_report_docx",
    "invalidity_report_markdown",
]
