"""Patent set topic clustering.

Embeds title+abstract per patent, grid-searches KMeans k (or DBSCAN eps) by
silhouette score, picks representative patents per cluster, and names each
cluster topic via a single LLM JSON call (falling back to pure-Python TF-IDF
top terms when no LLM is available).

sklearn / numpy / matplotlib are optional extras imported lazily::

    pip install 'patentkit[viz]'

:func:`top_terms_per_cluster` is dependency-free so topic naming stays
testable without sklearn.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

from patentkit.models import Patent

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with which said wherein thereof comprising method system
    apparatus device first second one more least each other than may can such
    using used use based include includes including data unit means plurality
    configured present invention according embodiment embodiments claim claims""".split()
)


def _require_viz():
    """Import numpy + sklearn or raise a helpful error."""
    try:
        import numpy as np
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError(
            "Clustering requires scikit-learn and numpy. "
            "Install them with: pip install 'patentkit[viz]'"
        ) from exc
    return np, KMeans, DBSCAN, PCA, silhouette_score


@dataclass
class ClusterResult:
    """Output of :func:`cluster_patents`.

    Attributes:
        labels: cluster id per input patent (-1 = DBSCAN noise).
        topics: cluster id -> short human-readable topic.
        silhouette: silhouette score of the chosen clustering (0.0 when undefined).
        representative: cluster id -> patent numbers nearest the cluster centroid.
        coords: optional 2D projection (one [x, y] per patent) for plotting.
    """

    labels: list[int]
    topics: dict[int, str]
    silhouette: float
    representative: dict[int, list[str]]
    coords: Optional[list[list[float]]] = None


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def top_terms_per_cluster(texts: Sequence[str], labels: Sequence[int],
                          top_n: int = 4) -> dict[int, list[str]]:
    """Pure-Python TF-IDF top terms per cluster (no sklearn needed).

    Treats each cluster's concatenated text as one document; IDF is computed
    across clusters so terms shared by every cluster are downweighted.

    Args:
        texts: one text per item.
        labels: cluster id per item (aligned with ``texts``).
        top_n: number of terms to return per cluster.

    Returns:
        cluster id -> top terms, highest TF-IDF first.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length")
    cluster_tf: dict[int, Counter] = {}
    for text, label in zip(texts, labels):
        cluster_tf.setdefault(label, Counter()).update(_tokenize(text))
    n_clusters = len(cluster_tf)
    df: Counter = Counter()
    for tf in cluster_tf.values():
        df.update(tf.keys())
    out: dict[int, list[str]] = {}
    for label, tf in cluster_tf.items():
        total = sum(tf.values()) or 1
        scored = {
            term: (count / total) * math.log(1 + n_clusters / df[term])
            for term, count in tf.items()
        }
        out[label] = [t for t, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    return out


def _patent_text(patent: Patent) -> str:
    return f"{patent.title or ''} {patent.abstract or ''}".strip() or str(patent.patent_number)


def _name_topics_with_llm(llm, representative_titles: dict[int, list[str]]) -> dict[int, str]:
    """Single JSON call mapping cluster id -> 3-6 word topic."""
    lines = []
    for cluster_id in sorted(representative_titles):
        titles = "; ".join(representative_titles[cluster_id]) or "(no titles)"
        lines.append(f"Cluster {cluster_id}: {titles}")
    prompt = (
        "Below are clusters of patents with representative titles. Name each "
        "cluster with a concise 3-6 word technology topic.\n\n"
        + "\n".join(lines)
        + '\n\nRespond with ONLY a JSON object mapping cluster id to topic, '
        'e.g. {"0": "wireless sensor power management"}.'
    )
    raw = llm.complete_json(prompt, max_tokens=1024)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object from the LLM, got {type(raw).__name__}")
    return {int(key): str(value) for key, value in raw.items()}


def cluster_patents(
    patents: Sequence[Patent],
    embeddings_provider,
    algorithm: str = "kmeans",
    n_clusters: int | None = None,
    llm=None,
) -> ClusterResult:
    """Cluster a patent set by title+abstract embeddings and name the topics.

    Args:
        patents: the patents to cluster (at least 4).
        embeddings_provider: any :class:`~patentkit.search.base.EmbeddingProvider`.
        algorithm: ``"kmeans"`` (silhouette grid search over k when
            ``n_clusters`` is None) or ``"dbscan"`` (silhouette grid search
            over an eps range, cosine metric).
        n_clusters: fixed k for kmeans; ignored for dbscan.
        llm: optional LLM used for topic naming; defaults to
            ``get_llm("low")``, with a TF-IDF term fallback when no LLM is
            usable.

    Returns:
        A :class:`ClusterResult` with labels, topics, silhouette score,
        representative patent numbers, and 2D plot coordinates.
    """
    np, KMeans, DBSCAN, PCA, silhouette_score = _require_viz()

    if len(patents) < 4:
        raise ValueError(f"Need at least 4 patents to cluster, got {len(patents)}")
    texts = [_patent_text(p) for p in patents]
    matrix = np.asarray(embeddings_provider.embed(texts), dtype=float)
    n = matrix.shape[0]

    # Optional PCA: denoise high-dimensional embeddings before clustering.
    if matrix.shape[1] > 50 and n > 10:
        matrix = PCA(n_components=min(50, n - 1)).fit_transform(matrix)

    if algorithm == "kmeans":
        labels, score = _fit_kmeans(np, KMeans, silhouette_score, matrix, n_clusters)
    elif algorithm == "dbscan":
        labels, score = _fit_dbscan(np, DBSCAN, silhouette_score, matrix)
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; use 'kmeans' or 'dbscan'")

    coords = PCA(n_components=2).fit_transform(matrix) if matrix.shape[1] > 2 else matrix[:, :2]

    representative = _representatives(np, matrix, labels, patents)
    representative_titles = {
        cluster_id: [
            (patents[i].title or str(patents[i].patent_number))
            for i in indices
        ]
        for cluster_id, indices in _representative_indices(np, matrix, labels).items()
    }

    topics = top_terms_joined(texts, labels)
    try:
        chosen_llm = llm
        if chosen_llm is None:
            from patentkit.llm import get_llm
            chosen_llm = get_llm("low")
        llm_topics = _name_topics_with_llm(chosen_llm, representative_titles)
        topics.update(llm_topics)
    except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to TF-IDF
        logger.warning("LLM topic naming unavailable (%s); using TF-IDF terms", exc)

    return ClusterResult(
        labels=[int(label) for label in labels],
        topics=topics,
        silhouette=float(score),
        representative=representative,
        coords=[[float(x), float(y)] for x, y in coords],
    )


def top_terms_joined(texts: Sequence[str], labels: Sequence[int]) -> dict[int, str]:
    """TF-IDF fallback topics: top terms joined into one string per cluster."""
    return {
        cluster_id: " / ".join(terms) if terms else "misc"
        for cluster_id, terms in top_terms_per_cluster(texts, labels).items()
    }


def _fit_kmeans(np, KMeans, silhouette_score, matrix, n_clusters: int | None):
    """KMeans with silhouette-scored grid search over k in 2..min(12, n//2)."""
    n = matrix.shape[0]
    if n_clusters is not None:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(matrix)
        score = silhouette_score(matrix, labels) if len(set(labels)) > 1 else 0.0
        return labels, score
    best_labels, best_score = None, -1.0
    for k in range(2, min(12, n // 2) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        logger.debug("kmeans k=%d silhouette=%.4f", k, score)
        if score > best_score:
            best_labels, best_score = labels, score
    if best_labels is None:  # degenerate data: everything identical
        best_labels, best_score = np.zeros(n, dtype=int), 0.0
    return best_labels, best_score


def _fit_dbscan(np, DBSCAN, silhouette_score, matrix):
    """DBSCAN (cosine) with silhouette-scored grid search over an eps range."""
    n = matrix.shape[0]
    eps_grid = np.unique(np.concatenate([
        np.logspace(np.log10(0.02), np.log10(0.7), num=12),
        np.linspace(0.05, 0.7, num=8),
    ]))
    min_samples_grid = [2, max(2, int(round(math.log(n))))]
    best_labels, best_score = None, -1.0
    for eps in eps_grid:
        for min_samples in dict.fromkeys(min_samples_grid):
            labels = DBSCAN(eps=float(eps), min_samples=min_samples, metric="cosine").fit_predict(matrix)
            clustered = labels[labels != -1]
            if len(set(clustered.tolist())) < 2:
                continue
            score = silhouette_score(matrix, labels)
            logger.debug("dbscan eps=%.3f min_samples=%d silhouette=%.4f", eps, min_samples, score)
            if score > best_score:
                best_labels, best_score = labels, score
    if best_labels is None:
        logger.warning("DBSCAN grid found no multi-cluster labeling; returning one cluster")
        best_labels, best_score = np.zeros(n, dtype=int), 0.0
    return best_labels, best_score


def _representative_indices(np, matrix, labels, per_cluster: int = 3) -> dict[int, list[int]]:
    """Indices of the patents nearest each cluster centroid."""
    out: dict[int, list[int]] = {}
    for cluster_id in sorted(set(int(label) for label in labels)):
        if cluster_id == -1:
            continue
        member_idx = np.where(np.asarray(labels) == cluster_id)[0]
        centroid = matrix[member_idx].mean(axis=0)
        distances = np.linalg.norm(matrix[member_idx] - centroid, axis=1)
        nearest = member_idx[np.argsort(distances)][:per_cluster]
        out[cluster_id] = [int(i) for i in nearest]
    return out


def _representatives(np, matrix, labels, patents) -> dict[int, list[str]]:
    return {
        cluster_id: [str(patents[i].patent_number) for i in indices]
        for cluster_id, indices in _representative_indices(np, matrix, labels).items()
    }


def plot_clusters(result: ClusterResult, patents: Sequence[Patent], out_path: str) -> str:
    """Scatter-plot a :class:`ClusterResult` (2D coords) and save to ``out_path``.

    Requires matplotlib (``pip install 'patentkit[viz]'``). Cluster topics are
    annotated at cluster centroids; DBSCAN noise (-1) is drawn in grey.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install 'patentkit[viz]'"
        ) from exc

    if result.coords is None:
        raise ValueError("ClusterResult has no 2D coords to plot")
    if len(result.coords) != len(patents):
        raise ValueError("coords and patents length mismatch")

    xs = [c[0] for c in result.coords]
    ys = [c[1] for c in result.coords]
    fig, ax = plt.subplots(figsize=(10, 7))
    cluster_ids = sorted(set(result.labels))
    cmap = plt.get_cmap("tab20")
    for i, cluster_id in enumerate(cluster_ids):
        member = [j for j, label in enumerate(result.labels) if label == cluster_id]
        color = "lightgrey" if cluster_id == -1 else cmap(i % 20)
        ax.scatter([xs[j] for j in member], [ys[j] for j in member],
                   color=color, s=30, alpha=0.8,
                   label="noise" if cluster_id == -1 else result.topics.get(cluster_id, str(cluster_id)))
        if cluster_id != -1 and member:
            cx = sum(xs[j] for j in member) / len(member)
            cy = sum(ys[j] for j in member) / len(member)
            ax.annotate(result.topics.get(cluster_id, str(cluster_id)), (cx, cy),
                        fontsize=8, fontweight="bold", ha="center")
    ax.set_title(f"Patent clusters (silhouette={result.silhouette:.3f})")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved cluster plot to %s", out_path)
    return out_path
