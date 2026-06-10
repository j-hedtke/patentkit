"""Line-number location in issued US patent PDFs via robust linear regression.

Issued US patents print the specification in two columns per page with line
number markers (5, 10, 15, ...) in a center gutter shared by both columns.
Text extraction of those markers is noisy: markers go missing, get OCR-misread
("15" -> "75"), or pick up stray digits. The legacy approach interpolated
between consecutive markers and snapped passages to the nearest point, which
amplifies single-marker errors. This module instead fits a least-squares
linear model ``line = a*y + b`` per page with iterative outlier rejection, so
a couple of bad markers cannot corrupt the mapping.

The regression (:func:`fit_line_model`), marker filtering
(:func:`filter_marker_candidates`), and citation formatting
(:func:`format_patent_citation`) are pure python — unit-testable with zero
optional dependencies. PDF text extraction (:func:`extract_patent_page_lines`)
needs ``pymupdf`` and passage matching (:func:`locate_passage`) additionally
needs ``rapidfuzz``; both are imported lazily from the ``pdf`` extra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "LineMarker",
    "LineNumberModel",
    "PdfTextLine",
    "PassageLocation",
    "fit_line_model",
    "filter_marker_candidates",
    "extract_patent_page_lines",
    "locate_passage",
    "format_patent_citation",
]


# ---------------------------------------------------------------------------
# Pure-python regression core
# ---------------------------------------------------------------------------


@dataclass
class LineMarker:
    """One detected gutter line-number marker: printed line number at y."""

    y: float
    line: int


@dataclass
class LineNumberModel:
    """Fitted linear map from page y-coordinate to printed line number."""

    slope: float
    intercept: float
    r2: float
    n_markers: int
    dropped: int

    def predict_float(self, y: float) -> float:
        """Un-rounded predicted line number at ``y``."""
        return self.slope * y + self.intercept

    def predict(self, y: float) -> int:
        """Predicted line number at ``y``, rounded and clamped to >= 1."""
        return max(1, round(self.predict_float(y)))


def _least_squares(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Ordinary least squares fit ``value = slope*y + intercept``."""
    n = len(points)
    mean_y = sum(p[0] for p in points) / n
    mean_v = sum(p[1] for p in points) / n
    var_y = sum((p[0] - mean_y) ** 2 for p in points)
    if var_y == 0:
        raise ValueError("Line markers must span distinct y positions to fit a line model")
    cov = sum((p[0] - mean_y) * (p[1] - mean_v) for p in points)
    slope = cov / var_y
    intercept = mean_v - slope * mean_y
    return slope, intercept


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def fit_line_model(markers: list[LineMarker], *, max_iterations: int = 3) -> LineNumberModel:
    """Fit ``line = a*y + b`` to gutter markers with robust outlier rejection.

    Procedure: least-squares fit, compute residuals (in line units), drop
    points whose absolute residual exceeds ``max(2*std(residuals), 0.5)``,
    refit; iterate up to ``max_iterations`` times, never dropping below two
    surviving points.

    Raises:
        ValueError: if fewer than 2 markers are supplied, or all markers
            share one y position.
    """
    if len(markers) < 2:
        raise ValueError(f"Need at least 2 line markers to fit a model, got {len(markers)}")

    points = [(float(m.y), float(m.line)) for m in markers]
    dropped = 0
    slope, intercept = _least_squares(points)
    for _ in range(max_iterations):
        residuals = [value - (slope * y + intercept) for y, value in points]
        threshold = max(2.0 * _std(residuals), 0.5)
        kept = [p for p, r in zip(points, residuals) if abs(r) <= threshold]
        if len(kept) == len(points) or len(kept) < 2:
            break
        dropped += len(points) - len(kept)
        points = kept
        slope, intercept = _least_squares(points)

    residuals = [value - (slope * y + intercept) for y, value in points]
    ss_res = sum(r * r for r in residuals)
    mean_v = sum(v for _, v in points) / len(points)
    ss_tot = sum((v - mean_v) ** 2 for _, v in points)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    if dropped:
        logger.debug("fit_line_model dropped %d outlier marker(s); r2=%.4f", dropped, r2)
    return LineNumberModel(
        slope=slope, intercept=intercept, r2=r2, n_markers=len(points), dropped=dropped
    )


def filter_marker_candidates(
    candidates: Iterable[tuple[float, float, str]], page_width: float
) -> list[LineMarker]:
    """Filter raw word candidates ``(x_center, y, text)`` down to gutter markers.

    A marker is a short numeric string whose value is divisible by 5 and lies
    in 5..100, located near the page midline OR within a consistent narrow
    x-band shared by the other numeric candidates (handles slightly off-center
    gutters). Pure python; shared by tests and the pymupdf extractor.
    """
    numeric: list[tuple[float, float, int]] = []
    for x, y, text in candidates:
        token = text.strip()
        if not token.isdigit() or len(token) > 3:
            continue
        value = int(token)
        if value % 5 != 0 or not (5 <= value <= 100):
            continue
        numeric.append((x, y, value))
    if not numeric:
        return []

    midline = page_width / 2.0
    near_mid = [c for c in numeric if abs(c[0] - midline) <= 0.12 * page_width]
    band_pool = near_mid or numeric
    xs = sorted(c[0] for c in band_pool)
    band_x = xs[len(xs) // 2]  # median x of the gutter band

    markers = [
        LineMarker(y=y, line=value)
        for x, y, value in numeric
        if abs(x - midline) <= 0.12 * page_width or abs(x - band_x) <= 0.03 * page_width
    ]
    markers.sort(key=lambda m: m.y)
    return markers


# ---------------------------------------------------------------------------
# PDF text extraction (pymupdf, lazy)
# ---------------------------------------------------------------------------


@dataclass
class PdfTextLine:
    """One extracted line of body text with its page coordinates."""

    text: str
    y: float
    x: float
    page: int  # 0-based page index in the PDF
    column: int  # 0 = left column, 1 = right column


@dataclass
class PassageLocation:
    """Where a passage sits in the printed column/line coordinate system."""

    page: int
    first_column: int
    first_line: int
    last_column: int
    last_line: int
    score: float


def _import_pymupdf():
    try:
        import fitz  # type: ignore[import-not-found]

        return fitz
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "pymupdf is required for patent PDF parsing; install with: pip install 'patentkit[pdf]'"
        ) from exc


def _import_rapidfuzz():
    try:
        from rapidfuzz import fuzz  # type: ignore[import-not-found]

        return fuzz
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "rapidfuzz is required for passage location; install with: pip install 'patentkit[pdf]'"
        ) from exc


@dataclass
class _PageData:
    page: int
    width: float
    lines: list[PdfTextLine]
    markers: list[LineMarker]


def _load_pages(pdf_path: str, pages: Optional[Sequence[int]] = None) -> list[_PageData]:
    """Extract body lines and gutter markers for each requested page."""
    fitz = _import_pymupdf()
    out: list[_PageData] = []
    with fitz.open(pdf_path) as doc:
        page_numbers = list(pages) if pages is not None else list(range(doc.page_count))
        for pno in page_numbers:
            page = doc[pno]
            width = float(page.rect.width)
            midline = width / 2.0
            words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)

            markers = filter_marker_candidates(
                (((w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0, w[4]) for w in words),
                width,
            )

            # Group words into visual lines keyed by (block, line); skip gutter markers.
            grouped: dict[tuple[int, int], list] = {}
            for w in words:
                x_center = (w[0] + w[2]) / 2.0
                token = w[4].strip()
                is_marker_like = (
                    token.isdigit()
                    and len(token) <= 3
                    and abs(x_center - midline) <= 0.12 * width
                )
                if is_marker_like:
                    continue
                grouped.setdefault((w[5], w[6]), []).append(w)

            lines: list[PdfTextLine] = []
            for ws in grouped.values():
                ws.sort(key=lambda w: w[0])
                text = " ".join(w[4] for w in ws).strip()
                if not text:
                    continue
                x0 = min(w[0] for w in ws)
                y = sum((w[1] + w[3]) / 2.0 for w in ws) / len(ws)
                x_mid = (x0 + max(w[2] for w in ws)) / 2.0
                column = 0 if x_mid < midline else 1
                lines.append(PdfTextLine(text=text, y=y, x=x0, page=pno, column=column))
            lines.sort(key=lambda ln: (ln.column, ln.y))
            out.append(_PageData(page=pno, width=width, lines=lines, markers=markers))
    return out


def extract_patent_page_lines(
    pdf_path: str, pages: Optional[Sequence[int]] = None
) -> list[PdfTextLine]:
    """Extract body-text lines (with coordinates) from an issued US patent PDF.

    Words are grouped into visual lines, classified into left/right columns by
    position relative to the page midline, and gutter line-number markers are
    excluded from the text. Requires ``pymupdf`` (``pip install 'patentkit[pdf]'``).

    Args:
        pdf_path: path to the patent PDF.
        pages: 0-based page indexes to extract; all pages if ``None``.
    """
    lines: list[PdfTextLine] = []
    for page_data in _load_pages(pdf_path, pages):
        lines.extend(page_data.lines)
    return lines


def page_markers(pdf_path: str, pages: Optional[Sequence[int]] = None) -> dict[int, list[LineMarker]]:
    """Detected gutter line-number markers per page (markers apply to both columns)."""
    return {pd.page: pd.markers for pd in _load_pages(pdf_path, pages)}


def locate_passage(
    pdf_path: str,
    passage: str,
    *,
    pages: Optional[Sequence[int]] = None,
    threshold: float = 80.0,
) -> Optional[PassageLocation]:
    """Fuzzy-locate ``passage`` in a patent PDF and map it to column/line numbers.

    The column texts are concatenated in reading order and matched with
    rapidfuzz partial alignment; matches scoring at or below ``threshold``
    return ``None``. The first/last matched lines' y-coordinates are then run
    through that page's fitted :class:`LineNumberModel` (markers in the center
    gutter are shared by both columns of a page).

    Printed column numbers are assigned sequentially over the scanned pages
    (two columns per page), so pass ``pages`` limited to the specification
    pages for accurate absolute column numbers.
    """
    fuzz = _import_rapidfuzz()
    page_data = _load_pages(pdf_path, pages)

    ordered: list[PdfTextLine] = []
    for pd in page_data:
        ordered.extend(pd.lines)

    spans: list[tuple[int, int, PdfTextLine]] = []
    parts: list[str] = []
    cursor = 0
    for line in ordered:
        start = cursor
        parts.append(line.text)
        cursor += len(line.text) + 1  # +1 for the join "\n"
        spans.append((start, cursor - 1, line))
    full_text = "\n".join(parts)
    if not full_text.strip():
        return None

    alignment = fuzz.partial_ratio_alignment(passage.lower(), full_text.lower())
    if alignment is None or alignment.score <= threshold:
        logger.debug("Passage not found above threshold %.1f", threshold)
        return None
    dest_start, dest_end = alignment.dest_start, max(alignment.dest_end - 1, alignment.dest_start)

    first_line = next((ln for s, e, ln in spans if dest_start < e), None)
    last_line = next((ln for s, e, ln in reversed(spans) if s <= dest_end), None)
    if first_line is None or last_line is None:
        return None

    models: dict[int, LineNumberModel] = {}
    page_rank = {pd.page: rank for rank, pd in enumerate(page_data)}
    markers_by_page = {pd.page: pd.markers for pd in page_data}

    def model_for(page: int) -> Optional[LineNumberModel]:
        if page not in models:
            try:
                models[page] = fit_line_model(markers_by_page.get(page, []))
            except ValueError as exc:
                logger.warning("Cannot fit line model for page %d: %s", page, exc)
                return None
        return models[page]

    first_model = model_for(first_line.page)
    last_model = model_for(last_line.page)
    if first_model is None or last_model is None:
        return None

    def column_number(line: PdfTextLine) -> int:
        return 2 * page_rank[line.page] + line.column + 1

    return PassageLocation(
        page=first_line.page,
        first_column=column_number(first_line),
        first_line=first_model.predict(first_line.y),
        last_column=column_number(last_line),
        last_line=last_model.predict(last_line.y),
        score=float(alignment.score),
    )


def format_patent_citation(loc: PassageLocation, patent_number: Optional[str] = None) -> str:
    """Format a :class:`PassageLocation` as a conventional patent citation.

    Examples: ``"col. 3, ll. 45-52"``, ``"col. 3, l. 45"``,
    ``"col. 3, l. 60 to col. 4, l. 5"`` (with optional patent number prefix).
    """
    if loc.first_column == loc.last_column:
        if loc.first_line == loc.last_line:
            body = f"col. {loc.first_column}, l. {loc.first_line}"
        else:
            body = f"col. {loc.first_column}, ll. {loc.first_line}-{loc.last_line}"
    else:
        body = (
            f"col. {loc.first_column}, l. {loc.first_line} to "
            f"col. {loc.last_column}, l. {loc.last_line}"
        )
    return f"{patent_number}, {body}" if patent_number else body
