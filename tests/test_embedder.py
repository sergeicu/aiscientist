# tests/test_embedder.py

import pytest
import numpy as np
from src.embeddings.embedder import ArticleEmbedder


@pytest.fixture
def sample_articles():
    return [
        {'pmid': '1', 'title': 'CAR-T therapy', 'abstract': 'CAR-T cells...'},
        {'pmid': '2', 'title': 'CRISPR editing', 'abstract': 'Gene editing...'}
    ]


def test_load_model():
    """Should load sentence-transformer model."""
    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')
    assert embedder.model is not None
    assert embedder.embedding_dim == 384


def test_embed_single_article(sample_articles):
    """Should generate embedding for one article."""
    embedder = ArticleEmbedder()
    embedding = embedder.embed_one(sample_articles[0])

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == embedder.embedding_dim


def test_embed_batch(sample_articles):
    """Should generate embeddings for multiple articles."""
    embedder = ArticleEmbedder()
    embeddings = embedder.embed_batch(sample_articles)

    assert embeddings.shape == (2, embedder.embedding_dim)


def test_normalize_embeddings(sample_articles):
    """Should normalize embeddings to unit length."""
    embedder = ArticleEmbedder()
    embeddings = embedder.embed_batch(sample_articles, normalize=True)

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)


def test_cache_embeddings(sample_articles, tmp_path):
    """Should cache embeddings to disk."""
    cache_path = tmp_path / "embeddings.npy"

    embedder = ArticleEmbedder()
    embeddings = embedder.embed_batch(sample_articles, cache_path=str(cache_path))

    assert cache_path.exists()
    loaded = np.load(cache_path)
    np.testing.assert_array_equal(embeddings, loaded)
