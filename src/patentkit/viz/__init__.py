"""Visualization and patent-set analysis (clustering, plotting).

Heavy dependencies (numpy, scikit-learn, matplotlib) are optional extras::

    pip install 'patentkit[viz]'
"""

from patentkit.viz.clustering import (
    ClusterResult,
    cluster_patents,
    plot_clusters,
    top_terms_per_cluster,
)

__all__ = [
    "ClusterResult",
    "cluster_patents",
    "plot_clusters",
    "top_terms_per_cluster",
]
