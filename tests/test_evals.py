"""Tests for eval metrics, datasets, and the harness (no network)."""

from __future__ import annotations

import json

import pytest

from patentkit.evals import (
    EvalRunner,
    QuerySet,
    UserEvalSetBuilder,
    average_precision,
    default_ipr_toy_dataset,
    load_queryset_jsonl,
    mean_recall_curve,
    mrr,
    recall_at_k,
    recall_curve,
    save_queryset_jsonl,
    searchfn_from_stores,
)
from patentkit.evals.datasets import IPR_TOY_PATH_ENV
from patentkit.models import Patent, PatentNumber
from patentkit.search.bm25 import BM25Store


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_kind_code_insensitive_matching(self):
        assert recall_at_k(["US7654321B2"], ["US7654321"], 1) == 1.0
        assert recall_at_k(["US7,654,321"], ["US7654321B1"], 1) == 1.0
        assert mrr(["US7654321B2"], ["US7654321"]) == 1.0

    def test_leading_zeros_ignored(self):
        assert recall_at_k(["US07654321"], ["US7654321"], 1) == 1.0

    def test_unparseable_strings_fall_back_to_exact(self):
        assert recall_at_k(["some-doc-id"], ["SOME-DOC-ID"], 1) == 1.0
        assert recall_at_k(["other-id"], ["SOME-DOC-ID"], 1) == 0.0

    def test_recall_at_k_hand_computed(self):
        preds = ["US1", "US2", "US3"]
        refs = ["US2", "US9"]
        assert recall_at_k(preds, refs, 1) == 0.0
        assert recall_at_k(preds, refs, 2) == 0.5
        assert recall_at_k(preds, refs, 10) == 0.5
        assert recall_at_k(preds, [], 5) == 0.0

    def test_duplicate_predictions_count_once(self):
        assert recall_at_k(["US2", "US2B1", "US2A"], ["US2", "US9"], 3) == 0.5

    def test_recall_curve(self):
        curve = recall_curve(["US2", "USX", "US9"], ["US2", "US9"], max_k=5)
        assert curve == [0.5, 0.5, 1.0, 1.0, 1.0]

    def test_mean_recall_curve_pads_with_last_value(self):
        assert mean_recall_curve([[1.0], [0.0, 0.5]]) == [0.5, 0.75]
        assert mean_recall_curve([]) == []

    def test_mrr_hand_computed(self):
        assert mrr(["miss", "US5", "US6"], ["US6", "US5"]) == pytest.approx(0.5)
        assert mrr(["miss"], ["US6"]) == 0.0

    def test_average_precision_hand_computed(self):
        # hits at ranks 1 and 3: (1/1 + 2/3) / 2
        ap = average_precision(["US1", "miss", "US2"], ["US1", "US2"])
        assert ap == pytest.approx((1.0 + 2 / 3) / 2)
        assert average_precision([], ["US1"]) == 0.0
        assert average_precision(["US1"], []) == 0.0


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_toy_dataset_loads_12_items(self):
        items = default_ipr_toy_dataset()
        assert len(items) == 12
        for item in items:
            assert item.query_patent.startswith("US")
            assert 2 <= len(item.references) <= 4
            assert item.metadata["toy"] is True
            assert item.metadata["proceeding"].startswith("IPR")
            assert item.claims

    def test_env_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom.jsonl"
        custom.write_text(json.dumps({
            "query_patent": "US1", "references": ["US2"], "metadata": {"toy": True},
        }) + "\n")
        monkeypatch.setenv(IPR_TOY_PATH_ENV, str(custom))
        items = default_ipr_toy_dataset()
        assert len(items) == 1
        assert items[0].query_patent == "US1"

    def test_jsonl_roundtrip(self, tmp_path):
        items = [
            QuerySet(query_patent="US100", claims=[1, 2], references=["US200", "US300"],
                     metadata={"k": "v"}),
            QuerySet(query_patent="US101", references=["US201"]),
        ]
        path = tmp_path / "qs.jsonl"
        assert save_queryset_jsonl(items, path) == 2
        loaded = load_queryset_jsonl(path)
        assert loaded == items

    def test_invalid_line_raises_with_location(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"query_patent": "US1", "references": []}\nnot json\n')
        with pytest.raises(ValueError, match="line 2"):
            load_queryset_jsonl(path)

    def test_user_eval_set_builder(self):
        builder = UserEvalSetBuilder()
        builder.add_judgment("US100", "US200", relevant=True)
        builder.add_judgment("US100", "US201", relevant=False)
        builder.add_judgment("US100", "US202", relevant=True)
        builder.add_judgment("US999", "US300", relevant=False)  # no relevant refs
        querysets = builder.to_querysets(metadata={"reviewer": "jh"})
        assert len(querysets) == 1
        [qs] = querysets
        assert qs.query_patent == "US100"
        assert qs.references == ["US200", "US202"]
        assert qs.metadata["rejected"] == ["US201"]
        assert qs.metadata["reviewer"] == "jh"
        assert qs.metadata["source"] == "user_judgments"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_dataset() -> list[QuerySet]:
    return [
        QuerySet(query_patent="US100", references=["US200", "US300"]),
        QuerySet(query_patent="US101", references=["US400"]),
    ]


class TestEvalRunner:
    def test_end_to_end_with_fake_search_fn(self, small_dataset, tmp_path):
        def search_fn(qs: QuerySet) -> list[str]:
            if qs.query_patent == "US100":
                # finds both refs (kind codes differ), refs at ranks 1 and 3
                return ["US200B1", "miss", "US300A"]
            return ["nope1", "nope2"]

        progress_calls: list[tuple] = []
        report = EvalRunner(search_fn, small_dataset, name="toy-run").run(
            max_k=50, progress=lambda done, total, q: progress_calls.append((done, total, q))
        )

        assert not report.errors
        assert len(report.rows) == 2
        assert report.aggregates["mean_recall@10"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert report.aggregates["mean_recall@50"] == pytest.approx(0.5)
        assert "mean_recall@100" not in report.aggregates  # capped by max_k
        assert report.aggregates["MRR"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        # query 1 AP = (1/1 + 2/3) / 2, query 2 AP = 0; MAP = mean
        assert report.aggregates["MAP"] == pytest.approx(((1.0 + 2 / 3) / 2) / 2)
        assert len(report.mean_curve) == 50
        assert progress_calls == [(1, 2, "US100"), (2, 2, "US101")]

        markdown = report.to_markdown()
        assert "mean_recall@10" in markdown
        assert "MRR" in markdown
        assert "MAP" in markdown
        assert "US100" in markdown

        out = tmp_path / "report.json"
        report.save_json(out)
        payload = json.loads(out.read_text())
        assert payload["name"] == "toy-run"
        assert payload["aggregates"]["mean_recall@10"] == pytest.approx(0.5)

    def test_per_query_errors_are_captured(self, small_dataset):
        def search_fn(qs: QuerySet) -> list[str]:
            if qs.query_patent == "US101":
                raise RuntimeError("backend exploded")
            return ["US200"]

        report = EvalRunner(search_fn, small_dataset).run(max_k=10)
        assert len(report.rows) == 1
        assert len(report.errors) == 1
        assert report.errors[0]["query_patent"] == "US101"
        assert "backend exploded" in report.errors[0]["error"]
        assert "backend exploded" in report.to_markdown()


class TestSearchFnFromStores:
    def test_runs_end_to_end_on_bm25(self):
        store = BM25Store()
        patents = [
            Patent(patent_number=PatentNumber.parse("US100"),
                   title="Neural network training",
                   abstract="Training neural networks with gradient descent."),
            Patent(patent_number=PatentNumber.parse("US200"),
                   title="Neural network accelerator",
                   abstract="Inference hardware for neural networks."),
            Patent(patent_number=PatentNumber.parse("US300"),
                   title="Gear shaft assembly",
                   abstract="Torque transmission through gears."),
        ]
        store.index(patents)
        dataset = [QuerySet(query_patent="US100", references=["US200"])]
        search_fn = searchfn_from_stores(store, limit=10)

        predictions = search_fn(dataset[0])
        assert "US100" not in predictions  # query patent excluded
        assert "US200" in predictions

        report = EvalRunner(search_fn, dataset).run(max_k=10)
        assert report.aggregates["mean_recall@10"] == 1.0

    def test_missing_query_patent_returns_empty(self):
        store = BM25Store()
        search_fn = searchfn_from_stores(store)
        assert search_fn(QuerySet(query_patent="US999", references=["US1"])) == []
