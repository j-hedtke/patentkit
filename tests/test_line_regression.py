"""Tests for the pure-python line-number regression model and citation
formatting (no pymupdf/rapidfuzz required)."""

import pytest

from patentkit.parsing.patent_pdf import (
    LineMarker,
    PassageLocation,
    filter_marker_candidates,
    fit_line_model,
    format_patent_citation,
)


def _perfect_markers(slope_y_per_line=3.0, y0=40.0, lines=range(5, 65, 5)):
    return [LineMarker(y=y0 + slope_y_per_line * line, line=line) for line in lines]


class TestFitLineModel:
    def test_perfect_markers_exact_recovery(self):
        markers = _perfect_markers()
        model = fit_line_model(markers)
        assert model.dropped == 0
        assert model.n_markers == len(markers)
        assert model.r2 == pytest.approx(1.0)
        assert model.slope == pytest.approx(1.0 / 3.0)
        for line in range(1, 68):
            assert model.predict(40.0 + 3.0 * line) == line

    def test_noisy_markers_with_gross_outliers_dropped(self):
        noise = [0.3, -0.2, 0.25, -0.3, 0.1, -0.15, 0.2, -0.25, 0.05, -0.1, 0.3, -0.05]
        markers = [
            LineMarker(y=40.0 + 3.0 * line + noise[i], line=line)
            for i, line in enumerate(range(5, 65, 5))
        ]
        # Two gross outliers: OCR misreads "15" -> "65" and "50" -> "20".
        markers[2] = LineMarker(y=markers[2].y, line=65)
        markers[9] = LineMarker(y=markers[9].y, line=20)

        model = fit_line_model(markers)
        assert model.dropped == 2
        assert model.n_markers == 10
        for line in (1, 7, 23, 41, 60, 67):
            predicted = model.predict(40.0 + 3.0 * line)
            assert abs(predicted - line) <= 1, f"line {line} predicted as {predicted}"

    def test_fewer_than_two_markers_raises(self):
        with pytest.raises(ValueError):
            fit_line_model([])
        with pytest.raises(ValueError):
            fit_line_model([LineMarker(y=100.0, line=5)])

    def test_degenerate_same_y_raises(self):
        with pytest.raises(ValueError):
            fit_line_model([LineMarker(y=100.0, line=5), LineMarker(y=100.0, line=10)])

    def test_predict_clamps_to_line_one(self):
        model = fit_line_model(_perfect_markers())
        # y far above the first printed line would extrapolate below 1
        assert model.predict(0.0) == 1
        assert model.predict(-500.0) == 1
        assert model.predict_float(0.0) < 1


class TestMarkerFiltering:
    def test_keeps_gutter_numbers_rejects_noise(self):
        width = 600.0
        candidates = [
            (300.0, 100.0, "5"),     # gutter marker
            (301.0, 160.0, "10"),    # gutter marker
            (299.0, 220.0, "15"),    # gutter marker
            (300.0, 280.0, "12"),    # not divisible by 5
            (300.0, 300.0, "1024"),  # too long
            (50.0, 120.0, "20"),     # body text number far from gutter
            (300.0, 340.0, "abc"),   # not numeric
        ]
        markers = filter_marker_candidates(candidates, width)
        assert [(m.line, m.y) for m in markers] == [(5, 100.0), (10, 160.0), (15, 220.0)]


class TestCitationFormatting:
    def test_line_range_same_column(self):
        loc = PassageLocation(
            page=2, first_column=3, first_line=45, last_column=3, last_line=52, score=95.0
        )
        assert format_patent_citation(loc) == "col. 3, ll. 45-52"

    def test_single_line(self):
        loc = PassageLocation(
            page=0, first_column=3, first_line=45, last_column=3, last_line=45, score=99.0
        )
        assert format_patent_citation(loc) == "col. 3, l. 45"

    def test_cross_column(self):
        loc = PassageLocation(
            page=1, first_column=3, first_line=60, last_column=4, last_line=5, score=88.0
        )
        assert format_patent_citation(loc) == "col. 3, l. 60 to col. 4, l. 5"

    def test_with_patent_number(self):
        loc = PassageLocation(
            page=2, first_column=3, first_line=45, last_column=3, last_line=52, score=95.0
        )
        assert format_patent_citation(loc, "US1234567") == "US1234567, col. 3, ll. 45-52"
