import pytest
import numpy as np
from src.visualization.embedder import ArticleEmbedder

# Check if model is available
def is_model_available():
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer('all-MiniLM-L6-v2')
        return True
    except Exception:
        return False

# Skip tests if model download fails
pytestmark = pytest.mark.skipif(
    not is_model_available(),
    reason="Model 'all-MiniLM-L6-v2' not available (download failed or no network)"
)

@pytest.fixture
def sample_articles():
    return [
        {
            'pmid': '123',
            'title': 'CAR-T therapy for leukemia',
            'abstract': 'This study investigates CAR-T cell therapy...'
        },
        {
            'pmid': '456',
            'title': 'Checkpoint inhibitors in melanoma',
            'abstract': 'We examine immune checkpoint blockade...'
        },
        {
            'pmid': '789',
            'title': 'CRISPR gene editing in sickle cell',
            'abstract': 'Gene editing using CRISPR-Cas9...'
        }
    ]

def test_create_embedder():
    """Should initialize embedder with model."""
    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')

    assert embedder.model is not None
    assert embedder.embedding_dim > 0

def test_embed_single_article(sample_articles):
    """Should generate embedding for single article."""
    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')

    embedding = embedder.embed_article(sample_articles[0])

    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1  # 1D vector
    assert embedding.shape[0] == embedder.embedding_dim

def test_embed_batch_articles(sample_articles):
    """Should generate embeddings for multiple articles."""
    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')

    embeddings = embedder.embed_articles(sample_articles)

    assert embeddings.shape[0] == 3  # 3 articles
    assert embeddings.shape[1] == embedder.embedding_dim

def test_combine_title_and_abstract(sample_articles):
    """Should combine title and abstract for embedding."""
    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')

    text = embedder.prepare_text(sample_articles[0])

    assert 'CAR-T therapy' in text
    assert 'investigates' in text

def test_handle_missing_abstract(sample_articles):
    """Should handle articles without abstracts."""
    article_no_abstract = {
        'pmid': '999',
        'title': 'Test Title',
        'abstract': None
    }

    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')
    embedding = embedder.embed_article(article_no_abstract)

    assert embedding is not None
    assert embedding.shape[0] == embedder.embedding_dim

def test_normalize_embeddings(sample_articles):
    """Should normalize embeddings to unit length."""
    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')

    embeddings = embedder.embed_articles(
        sample_articles,
        normalize=True
    )

    # Check if normalized (L2 norm should be ~1)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01)

def test_caching_embeddings(sample_articles, tmp_path):
    """Should cache embeddings to disk."""
    cache_path = tmp_path / "embeddings.npy"

    embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')

    # Generate and cache
    embeddings = embedder.embed_articles(
        sample_articles,
        cache_path=str(cache_path)
    )

    # Check cache file exists
    assert cache_path.exists()

    # Load from cache
    cached = embedder.load_embeddings(str(cache_path))

    np.testing.assert_array_equal(embeddings, cached)
