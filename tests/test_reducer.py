import pytest
import numpy as np
from src.visualization.reducer import UMAPReducer

@pytest.fixture
def sample_embeddings():
    """Create sample high-dimensional embeddings."""
    np.random.seed(42)
    # 100 samples, 384 dimensions
    return np.random.randn(100, 384)

def test_create_reducer():
    """Should initialize UMAP reducer."""
    reducer = UMAPReducer(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1
    )

    assert reducer.n_components == 2
    assert reducer.n_neighbors == 15

def test_reduce_to_2d(sample_embeddings):
    """Should reduce to 2D."""
    reducer = UMAPReducer(n_components=2)

    reduced = reducer.fit_transform(sample_embeddings)

    assert reduced.shape[0] == 100  # Same number of samples
    assert reduced.shape[1] == 2    # 2 dimensions

def test_reduce_to_3d(sample_embeddings):
    """Should reduce to 3D."""
    reducer = UMAPReducer(n_components=3)

    reduced = reducer.fit_transform(sample_embeddings)

    assert reduced.shape[1] == 3  # 3 dimensions

def test_reproducible_results(sample_embeddings):
    """Should produce reproducible results with same seed."""
    reducer1 = UMAPReducer(n_components=2, random_state=42)
    reducer2 = UMAPReducer(n_components=2, random_state=42)

    reduced1 = reducer1.fit_transform(sample_embeddings)
    reduced2 = reducer2.fit_transform(sample_embeddings)

    np.testing.assert_array_almost_equal(reduced1, reduced2)

def test_transform_new_data(sample_embeddings):
    """Should transform new data using fitted reducer."""
    train_data = sample_embeddings[:80]
    test_data = sample_embeddings[80:]

    reducer = UMAPReducer(n_components=2)
    reducer.fit(train_data)

    # Transform new data
    reduced_test = reducer.transform(test_data)

    assert reduced_test.shape[0] == 20
    assert reduced_test.shape[1] == 2

def test_save_and_load_reducer(sample_embeddings, tmp_path):
    """Should save and load reducer."""
    save_path = tmp_path / "reducer.pkl"

    # Fit and save
    reducer = UMAPReducer(n_components=2)
    reducer.fit(sample_embeddings)
    reducer.save(str(save_path))

    # Load
    loaded_reducer = UMAPReducer.load(str(save_path))

    # Transform with loaded reducer
    reduced_original = reducer.transform(sample_embeddings[:10])
    reduced_loaded = loaded_reducer.transform(sample_embeddings[:10])

    np.testing.assert_array_almost_equal(reduced_original, reduced_loaded)
