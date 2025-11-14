# Task 8: Embedding Generation & Vector Store Setup - TDD Implementation

## Executive Summary

Implement a production-ready embedding generation pipeline and vector store (ChromaDB) for semantic search over research articles. This creates the foundation for clustering, similarity search, and semantic analysis.

**Key Requirements:**
- Generate semantic embeddings using Sentence-BERT models
- Store embeddings in ChromaDB with metadata
- Implement semantic search functionality
- Support batch processing for large corpora
- Cache embeddings to disk
- Follow Test-Driven Development (TDD) principles

**Note**: This task can run **independently in parallel** with other tasks.

## Background & Context

### Why Embeddings?

**Problem**: Text is unstructured, high-dimensional, sparse
**Solution**: Convert to dense semantic vectors (embeddings)

**Benefits**:
- **Semantic Search**: Find similar articles by meaning, not keywords
- **Clustering**: Group similar articles together
- **Visualization**: Reduce dimensions for plotting
- **Transfer Learning**: Pre-trained models understand language

### Sentence-BERT Models

Popular models for scientific text:
- `all-MiniLM-L6-v2`: Fast, 384-dim (good for testing)
- `all-mpnet-base-v2`: Better quality, 768-dim (recommended)
- `allenai/specter`: Scientific papers, 768-dim (domain-specific)
- `sentence-transformers/allenai-specter2`: Latest scientific model

### ChromaDB

Open-source vector database:
- Persistent storage
- Metadata filtering
- Built-in similarity search
- No server required (embedded)
- Scales to millions of vectors

## Technical Architecture

```
src/embeddings/
├── __init__.py
├── embedder.py         # Embedding generation
├── vector_store.py     # ChromaDB operations
└── cache.py            # Disk caching

tests/
├── __init__.py
├── test_embedder.py
├── test_vector_store.py
└── test_cache.py
```

## TDD Implementation Plan

### Phase 1: Embedder (TDD)

#### Test Cases

```python
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
```

#### Implementation

```python
# src/embeddings/embedder.py

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path

class ArticleEmbedder:
    """
    Generate semantic embeddings for research articles.

    Example:
        >>> embedder = ArticleEmbedder(model_name='all-mpnet-base-v2')
        >>> embeddings = embedder.embed_batch(articles)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✓ Model loaded ({self.embedding_dim}-dim)")

    def prepare_text(self, article: Dict) -> str:
        """Combine title and abstract."""
        title = article.get('title', '')
        abstract = article.get('abstract', '')
        return f"{title}. {abstract}".strip() if abstract else title

    def embed_one(self, article: Dict, normalize: bool = False) -> np.ndarray:
        """Generate embedding for single article."""
        text = self.prepare_text(article)
        return self.model.encode(text, normalize_embeddings=normalize)

    def embed_batch(
        self,
        articles: List[Dict],
        batch_size: int = 32,
        normalize: bool = False,
        show_progress: bool = True,
        cache_path: Optional[str] = None
    ) -> np.ndarray:
        """Generate embeddings for multiple articles."""
        logger.info(f"Embedding {len(articles)} articles...")

        texts = [self.prepare_text(a) for a in articles]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"✓ Cached to {cache_path}")

        return embeddings
```

### Phase 2: Vector Store (TDD)

#### Test Cases

```python
# tests/test_vector_store.py

import pytest
from src.embeddings.vector_store import VectorStore

@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path / "chroma"))

@pytest.fixture
def sample_data():
    import numpy as np
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
```

#### Implementation

```python
# src/embeddings/vector_store.py

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from loguru import logger

class VectorStore:
    """
    Manage vector embeddings in ChromaDB.

    Example:
        >>> store = VectorStore()
        >>> store.add_embeddings(embeddings, ids, metadatas)
        >>> results = store.query(query_embedding, n_results=10)
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "articles"
    ):
        logger.info(f"Initializing ChromaDB at {persist_directory}")

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection_name = collection_name
        self.collection = self.get_or_create_collection(collection_name)

        logger.info(f"✓ Collection '{collection_name}' ready")

    def get_or_create_collection(self, name: str):
        """Get or create collection."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"description": "Research article embeddings"}
        )

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: List[Dict],
        batch_size: int = 5000
    ):
        """Add embeddings to collection."""
        logger.info(f"Adding {len(embeddings)} embeddings...")

        # ChromaDB batch limit
        for i in range(0, len(embeddings), batch_size):
            end = min(i + batch_size, len(embeddings))

            self.collection.add(
                embeddings=embeddings[i:end],
                ids=ids[i:end],
                metadatas=metadatas[i:end]
            )

            logger.debug(f"Added batch {i//batch_size + 1}")

        logger.info(f"✓ Added {len(embeddings)} embeddings")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:
        """Query similar embeddings."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        return {
            'ids': results['ids'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0]
        }

    def count(self) -> int:
        """Count embeddings in collection."""
        return self.collection.count()

    def delete(self, ids: List[str]):
        """Delete embeddings by ID."""
        self.collection.delete(ids=ids)
```

## Usage Examples

### Complete Pipeline

```python
from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import VectorStore
import json

# Load articles
with open('articles.json') as f:
    articles = json.load(f)

# Generate embeddings
embedder = ArticleEmbedder(model_name='all-mpnet-base-v2')
embeddings = embedder.embed_batch(articles)

# Store in ChromaDB
store = VectorStore()
store.add_embeddings(
    embeddings=embeddings.tolist(),
    ids=[a['pmid'] for a in articles],
    metadatas=[
        {'pmid': a['pmid'], 'title': a['title'], 'year': a['year']}
        for a in articles
    ]
)

# Semantic search
query_text = "CAR-T therapy for leukemia"
query_embedding = embedder.embed_one({'title': query_text, 'abstract': ''})

results = store.query(query_embedding, n_results=10)
for i, (id, distance) in enumerate(zip(results['ids'], results['distances'])):
    print(f"{i+1}. {id}: {distance:.3f}")
```

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Generate embeddings for articles
3. Store embeddings in ChromaDB
4. Semantic search working
5. Metadata filtering
6. Batch processing

✅ **Should Have:**
7. Caching to disk
8. Progress bars
9. Multiple model support
10. Error handling

## Running Tests

```bash
pip install sentence-transformers chromadb pytest numpy

pytest tests/test_embedder.py -v
pytest tests/test_vector_store.py -v
```

---

**Task completion**: When all tests pass, embeddings generate correctly, ChromaDB stores them, and semantic search returns relevant results.
