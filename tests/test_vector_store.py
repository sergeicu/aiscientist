# tests/test_vector_store.py

import pytest
import numpy as np
from src.embeddings.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path / "chroma"))


@pytest.fixture
def sample_data():
    return {
        'embeddings': np.random.randn(5, 384).tolist(),
        'ids': ['1', '2', '3', '4', '5'],
        'metadatas': [
            {'pmid': '1', 'title': 'Article 1', 'year': '2023'},
            {'pmid': '2', 'title': 'Article 2', 'year': '2023'},
            {'pmid': '3', 'title': 'Article 3', 'year': '2022'},
            {'pmid': '4', 'title': 'Article 4', 'year': '2022'},
            {'pmid': '5', 'title': 'Article 5', 'year': '2021'}
        ]
    }


def test_create_collection(store):
    """Should create collection."""
    collection = store.get_or_create_collection('test')
    assert collection is not None


def test_add_embeddings(store, sample_data):
    """Should add embeddings to collection."""
    store.add_embeddings(
        embeddings=sample_data['embeddings'],
        ids=sample_data['ids'],
        metadatas=sample_data['metadatas']
    )

    count = store.count()
    assert count == 5


def test_query_similar(store, sample_data):
    """Should query similar embeddings."""
    store.add_embeddings(**sample_data)

    # Query with first embedding
    results = store.query(
        query_embedding=sample_data['embeddings'][0],
        n_results=3
    )

    assert len(results['ids']) == 3
    assert results['ids'][0] == '1'  # Should return itself first


def test_filter_by_metadata(store, sample_data):
    """Should filter results by metadata."""
    store.add_embeddings(**sample_data)

    results = store.query(
        query_embedding=sample_data['embeddings'][0],
        n_results=10,
        where={'year': '2023'}
    )

    # Should only return 2023 articles
    assert all(m['year'] == '2023' for m in results['metadatas'])


def test_delete_by_ids(store, sample_data):
    """Should delete embeddings by ID."""
    store.add_embeddings(**sample_data)

    store.delete(ids=['1', '2'])

    assert store.count() == 3
