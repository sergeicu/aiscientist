import numpy as np
import hdbscan
from typing import Dict, List, Optional
from loguru import logger

class HDBSCANClusterer:
    """
    Cluster reduced embeddings using HDBSCAN.

    HDBSCAN automatically determines number of clusters
    and detects outliers/noise.

    Args:
        min_cluster_size: Minimum samples in cluster
        min_samples: Minimum samples for core points
        metric: Distance metric

    Example:
        >>> clusterer = HDBSCANClusterer(min_cluster_size=50)
        >>> labels = clusterer.fit_predict(reduced_embeddings)
    """

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        metric: str = 'euclidean'
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method='eom',
            prediction_data=True
        )

    def fit(self, data: np.ndarray):
        """Fit clusterer on data."""
        logger.info(f"Fitting HDBSCAN on {data.shape[0]} points...")

        self.clusterer.fit(data)

        logger.info("✓ HDBSCAN fitted")
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        # HDBSCAN doesn't have native predict, use approximate_predict
        labels, strengths = hdbscan.approximate_predict(self.clusterer, data)
        return labels

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit clusterer and predict labels.

        Args:
            data: Reduced embeddings

        Returns:
            Cluster labels (N,)
            -1 indicates noise/outliers
        """
        logger.info(f"Clustering {data.shape[0]} points...")

        labels = self.clusterer.fit_predict(data)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(
            f"✓ Clustering complete: {n_clusters} clusters, "
            f"{n_noise} outliers"
        )

        return labels

    def get_cluster_stats(self, labels: np.ndarray) -> Dict:
        """
        Calculate cluster statistics.

        Args:
            labels: Cluster labels

        Returns:
            Statistics dictionary
        """
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        # Cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = list(labels).count(label)

        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': n_noise / len(labels) * 100,
            'cluster_sizes': cluster_sizes,
            'largest_cluster': max(cluster_sizes.values()) if cluster_sizes else 0,
            'smallest_cluster': min(cluster_sizes.values()) if cluster_sizes else 0
        }
