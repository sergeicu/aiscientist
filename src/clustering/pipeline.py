# src/clustering/pipeline.py

import numpy as np
import umap
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple, Dict
from loguru import logger
import pickle


class ClusteringPipeline:
    """
    Complete clustering pipeline: UMAP + HDBSCAN.

    Example:
        >>> pipeline = ClusteringPipeline(n_components=2, min_cluster_size=50)
        >>> labels, reduced = pipeline.fit_predict(embeddings)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 50,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state

        # Initialize models
        self.umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=random_state
        )

        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=10,
            metric='euclidean'
        )

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensions with UMAP."""
        logger.info(f"UMAP: {embeddings.shape} → ({embeddings.shape[0]}, {self.n_components})")
        reduced = self.umap_model.fit_transform(embeddings)
        logger.info("✓ UMAP complete")
        return reduced

    def cluster(self, reduced: np.ndarray) -> np.ndarray:
        """Cluster with HDBSCAN."""
        logger.info(f"Clustering {reduced.shape[0]} points...")
        labels = self.hdbscan_model.fit_predict(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"✓ Found {n_clusters} clusters, {n_noise} outliers")
        return labels

    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit pipeline and return labels + reduced embeddings."""
        reduced = self.reduce_dimensions(embeddings)
        labels = self.cluster(reduced)
        return labels, reduced

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted UMAP."""
        return self.umap_model.transform(embeddings)

    def calculate_metrics(self, reduced: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate clustering quality metrics."""
        # Filter out noise points for metrics
        mask = labels != -1
        if mask.sum() < 2:
            return {'error': 'Not enough non-noise points'}

        reduced_filtered = reduced[mask]
        labels_filtered = labels[mask]

        metrics = {
            'n_clusters': len(set(labels_filtered)),
            'n_noise': list(labels).count(-1),
            'noise_percentage': list(labels).count(-1) / len(labels) * 100
        }

        # Silhouette score (only if multiple clusters)
        if len(set(labels_filtered)) > 1:
            metrics['silhouette_score'] = silhouette_score(reduced_filtered, labels_filtered)
            metrics['davies_bouldin_score'] = davies_bouldin_score(reduced_filtered, labels_filtered)

        return metrics

    def save(self, path: str):
        """Save fitted pipeline."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"✓ Saved pipeline to {path}")

    @staticmethod
    def load(path: str) -> 'ClusteringPipeline':
        """Load fitted pipeline."""
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        logger.info(f"✓ Loaded pipeline from {path}")
        return pipeline
