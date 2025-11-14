# Task 9: UMAP + HDBSCAN Clustering Pipeline - TDD Implementation

## Executive Summary

Implement dimensionality reduction (UMAP) and density-based clustering (HDBSCAN) pipeline for research articles. This creates interpretable clusters and 2D/3D visualizations of the research landscape.

**Key Requirements:**
- UMAP dimensionality reduction (768-dim → 2D/3D)
- HDBSCAN clustering (automatic cluster detection)
- Cluster quality metrics
- Save/load fitted models
- Generate cluster assignments
- Follow TDD principles

**Dependencies**: Requires Task 8 (embeddings) to provide input data.

## Why UMAP + HDBSCAN?

**UMAP**: Preserves local and global structure, faster than t-SNE
**HDBSCAN**: Automatically determines number of clusters, detects outliers

## TDD Implementation

### Test Cases

```python
# tests/test_clustering_pipeline.py

import pytest
import numpy as np
from src.clustering.pipeline import ClusteringPipeline

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings with 3 clusters."""
    np.random.seed(42)
    cluster1 = np.random.randn(100, 384) + [0]*384
    cluster2 = np.random.randn(100, 384) + [5]*384
    cluster3 = np.random.randn(100, 384) + [10]*384
    return np.vstack([cluster1, cluster2, cluster3])

def test_create_pipeline():
    """Should initialize pipeline."""
    pipeline = ClusteringPipeline(n_components=2, min_cluster_size=10)
    assert pipeline.n_components == 2

def test_reduce_dimensions(sample_embeddings):
    """Should reduce to 2D."""
    pipeline = ClusteringPipeline(n_components=2)
    reduced = pipeline.reduce_dimensions(sample_embeddings)

    assert reduced.shape == (300, 2)

def test_cluster_reduced(sample_embeddings):
    """Should cluster reduced embeddings."""
    pipeline = ClusteringPipeline(n_components=2, min_cluster_size=10)
    reduced = pipeline.reduce_dimensions(sample_embeddings)
    labels = pipeline.cluster(reduced)

    assert len(labels) == 300
    assert len(set(labels)) >= 2  # At least 2 clusters

def test_fit_predict(sample_embeddings):
    """Should fit and predict in one step."""
    pipeline = ClusteringPipeline()
    labels, reduced = pipeline.fit_predict(sample_embeddings)

    assert len(labels) == 300
    assert reduced.shape[0] == 300

def test_calculate_metrics(sample_embeddings):
    """Should calculate clustering quality metrics."""
    pipeline = ClusteringPipeline()
    labels, reduced = pipeline.fit_predict(sample_embeddings)

    metrics = pipeline.calculate_metrics(reduced, labels)

    assert 'silhouette_score' in metrics
    assert 'n_clusters' in metrics
    assert 'n_noise' in metrics

def test_save_load_pipeline(sample_embeddings, tmp_path):
    """Should save and load fitted pipeline."""
    pipeline = ClusteringPipeline()
    pipeline.fit_predict(sample_embeddings[:200])

    save_path = tmp_path / "pipeline.pkl"
    pipeline.save(str(save_path))

    loaded = ClusteringPipeline.load(str(save_path))

    # Transform new data
    new_reduced = loaded.transform(sample_embeddings[200:])
    assert new_reduced.shape[1] == pipeline.n_components
```

### Implementation

```python
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
```

## Usage Example

```python
from src.embeddings.embedder import ArticleEmbedder
from src.clustering.pipeline import ClusteringPipeline
import json
import pandas as pd

# Load articles and generate embeddings
with open('articles.json') as f:
    articles = json.load(f)

embedder = ArticleEmbedder()
embeddings = embedder.embed_batch(articles)

# Cluster
pipeline = ClusteringPipeline(n_components=2, min_cluster_size=50)
labels, reduced = pipeline.fit_predict(embeddings)

# Calculate metrics
metrics = pipeline.calculate_metrics(reduced, labels)
print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
print(f"Clusters: {metrics['n_clusters']}")

# Save results
df = pd.DataFrame({
    'pmid': [a['pmid'] for a in articles],
    'x': reduced[:, 0],
    'y': reduced[:, 1],
    'cluster': labels
})
df.to_csv('cluster_assignments.csv', index=False)
```

## Success Criteria

✅ Tests pass (>90% coverage)
✅ UMAP reduces dimensions correctly
✅ HDBSCAN detects clusters
✅ Quality metrics calculated
✅ Pipeline can be saved/loaded

## Running Tests

```bash
pip install umap-learn hdbscan scikit-learn pytest

pytest tests/test_clustering_pipeline.py -v
```

---

**Task completion**: When all tests pass and clustering produces interpretable results with good quality metrics.
