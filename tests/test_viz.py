"""Tests for clustering: pure TF-IDF fallback always; sklearn smoke when available."""

from __future__ import annotations

import pytest

from patentkit.models import Patent, PatentNumber
from patentkit.viz import cluster_patents, top_terms_per_cluster

NEURAL_TEXTS = [
    "Neural network training with deep learning and gradient descent optimization",
    "Deep neural network layers learn features for image classification training",
    "Training neural networks using backpropagation and learning rate schedules",
]
GEAR_TEXTS = [
    "Gear shaft bearing assembly for torque transmission in vehicle drivetrain",
    "Helical gear and shaft coupling with bearing lubrication for torque transfer",
    "Transmission gearbox with planetary gears, shafts and roller bearings",
]


class TestTopTermsPerCluster:
    def test_distinguishes_clusters(self):
        texts = NEURAL_TEXTS + GEAR_TEXTS
        labels = [0, 0, 0, 1, 1, 1]
        terms = top_terms_per_cluster(texts, labels, top_n=4)
        assert set(terms) == {0, 1}
        assert any(t in ("neural", "network", "training", "learning") for t in terms[0])
        assert any(t in ("gear", "shaft", "bearing", "torque") for t in terms[1])
        assert not set(terms[0]) & set(terms[1])

    def test_shared_terms_downweighted(self):
        texts = ["common neural alpha", "common neural alpha", "common gear beta", "common gear beta"]
        labels = [0, 0, 1, 1]
        terms = top_terms_per_cluster(texts, labels, top_n=1)
        # "common" appears in both clusters so cluster-specific terms win
        assert terms[0] != ["common"]
        assert terms[1] != ["common"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            top_terms_per_cluster(["a"], [0, 1])


def make_patent(i: int, title: str, abstract: str) -> Patent:
    return Patent(
        patent_number=PatentNumber.parse(f"US{9000000 + i}"),
        title=title,
        abstract=abstract,
    )


def two_topic_patents() -> list[Patent]:
    patents = []
    for i, text in enumerate(NEURAL_TEXTS + NEURAL_TEXTS):
        patents.append(make_patent(i, f"Neural patent {i}", text))
    for i, text in enumerate(GEAR_TEXTS + GEAR_TEXTS):
        patents.append(make_patent(100 + i, f"Gear patent {i}", text))
    return patents


class TestClusterPatents:
    def test_helpful_error_without_sklearn(self):
        try:
            import sklearn  # noqa: F401
            pytest.skip("sklearn installed; ImportError path not reachable")
        except ImportError:
            pass
        from patentkit.search.vector import HashingEmbeddings
        with pytest.raises(ImportError, match=r"patentkit\[viz\]"):
            cluster_patents(two_topic_patents(), HashingEmbeddings(32))

    def test_kmeans_smoke(self):
        pytest.importorskip("sklearn")
        from patentkit.search.vector import HashingEmbeddings
        from tests.fakes import FakeLLM

        patents = two_topic_patents()
        llm = FakeLLM(responses=[{"0": "neural network training", "1": "gear transmission"}])
        result = cluster_patents(
            patents, HashingEmbeddings(dimensions=64), algorithm="kmeans", llm=llm
        )

        assert len(result.labels) == len(patents)
        assert isinstance(result.silhouette, float)
        cluster_ids = {label for label in result.labels if label != -1}
        assert len(cluster_ids) >= 2
        assert set(result.topics) >= cluster_ids
        assert all(result.representative[c] for c in cluster_ids)
        assert result.coords is not None and len(result.coords) == len(patents)
        assert all(len(point) == 2 for point in result.coords)

    def test_dbscan_smoke(self):
        pytest.importorskip("sklearn")
        from patentkit.search.vector import HashingEmbeddings
        from tests.fakes import FakeLLM

        result = cluster_patents(
            two_topic_patents(), HashingEmbeddings(dimensions=64),
            algorithm="dbscan", llm=FakeLLM(default="{}"),
        )
        assert len(result.labels) == 12
        # topics fall back to TF-IDF terms when the LLM returns nothing useful
        assert result.topics

    def test_unknown_algorithm_raises(self):
        pytest.importorskip("sklearn")
        from patentkit.search.vector import HashingEmbeddings
        with pytest.raises(ValueError, match="algorithm"):
            cluster_patents(two_topic_patents(), HashingEmbeddings(32), algorithm="bogus")
