import pytest
import numpy as np
from src.visualization.clusterer import HDBSCANClusterer

@pytest.fixture
def sample_2d_data():
    """Create sample 2D data with clear clusters."""
    np.random.seed(42)

    # Three clusters
    cluster1 = np.random.randn(30, 2) + [0, 0]
    cluster2 = np.random.randn(30, 2) + [5, 5]
    cluster3 = np.random.randn(30, 2) + [10, 0]

    return np.vstack([cluster1, cluster2, cluster3])

def test_create_clusterer():
    """Should initialize HDBSCAN clusterer."""
    clusterer = HDBSCANClusterer(min_cluster_size=10)

    assert clusterer.min_cluster_size == 10

def test_fit_predict(sample_2d_data):
    """Should fit and predict cluster labels."""
    clusterer = HDBSCANClusterer(min_cluster_size=10)

    labels = clusterer.fit_predict(sample_2d_data)

    assert len(labels) == 90  # All samples
    assert len(set(labels)) >= 2  # At least 2 clusters (might include -1 for noise)

def test_noise_detection(sample_2d_data):
    """Should detect noise points (label -1)."""
    # Add noise points
    noise = np.random.randn(10, 2) * 20  # Far from clusters
    data_with_noise = np.vstack([sample_2d_data, noise])

    clusterer = HDBSCANClusterer(min_cluster_size=10)
    labels = clusterer.fit_predict(data_with_noise)

    # Should have some noise points
    assert -1 in labels

def test_get_cluster_stats(sample_2d_data):
    """Should calculate cluster statistics."""
    clusterer = HDBSCANClusterer(min_cluster_size=10)
    labels = clusterer.fit_predict(sample_2d_data)

    stats = clusterer.get_cluster_stats(labels)

    assert 'n_clusters' in stats
    assert 'n_noise' in stats
    assert 'cluster_sizes' in stats

def test_assign_to_existing_clusters(sample_2d_data):
    """Should assign new points to existing clusters."""
    train_data = sample_2d_data[:80]
    test_data = sample_2d_data[80:]

    clusterer = HDBSCANClusterer(min_cluster_size=10)
    clusterer.fit(train_data)

    # Predict for new data
    test_labels = clusterer.predict(test_data)

    assert len(test_labels) == 10
