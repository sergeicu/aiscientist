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
