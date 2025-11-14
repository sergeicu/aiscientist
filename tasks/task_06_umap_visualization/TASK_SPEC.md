# Task 6: UMAP Clustering Visualization Dashboard - TDD Implementation

## Executive Summary

Implement an interactive visualization dashboard for exploring research article clusters using UMAP dimensionality reduction. This creates 2D/3D scatter plots where similar articles appear close together, enabling discovery of research themes and trends.

**Key Requirements:**
- Generate embeddings for article abstracts using sentence transformers
- Reduce high-dimensional embeddings to 2D/3D using UMAP
- Cluster articles using HDBSCAN
- Create interactive Plotly scatter plots with hover information
- Build Streamlit dashboard with filters and search
- Support exporting visualizations and cluster assignments
- Follow Test-Driven Development (TDD) principles

## Background & Context

### Why UMAP Visualization?

**Research Discovery Benefits:**
- **Visual Exploration**: See research landscape at a glance
- **Cluster Identification**: Discover natural research groupings
- **Trend Spotting**: Identify emerging topics
- **Outlier Detection**: Find unique/novel research
- **Quality Check**: Verify clustering makes semantic sense

### UMAP (Uniform Manifold Approximation and Projection)

UMAP is superior to t-SNE for large datasets:
- **Faster**: Scales to millions of points
- **Global Structure**: Preserves both local and global relationships
- **Deterministic**: Reproducible results
- **Flexible**: Works in 2D and 3D

### Typical Workflow

```
1. Articles → 2. Embeddings → 3. UMAP → 4. Clustering → 5. Visualization
   (text)      (768-dim)      (2D/3D)    (labels)      (interactive plot)
```

## Technical Architecture

### Module Structure

```
src/visualization/
├── __init__.py
├── embedder.py          # Generate embeddings
├── reducer.py           # UMAP dimension reduction
├── clusterer.py         # HDBSCAN clustering
├── plotter.py           # Plotly visualizations
└── dashboard.py         # Streamlit app

tests/
├── __init__.py
├── test_embedder.py
├── test_reducer.py
├── test_clusterer.py
├── test_plotter.py
└── fixtures/
    └── sample_articles.json
```

### Dependencies

```python
# Required packages
sentence-transformers>=2.2  # Embeddings
umap-learn>=0.5            # Dimensionality reduction
hdbscan>=0.8               # Clustering
plotly>=5.17               # Interactive plots
streamlit>=1.28            # Dashboard framework
pandas>=2.0                # Data manipulation
numpy>=1.24                # Numerical operations
scikit-learn>=1.3          # Additional ML utilities
pydantic>=2.0              # Data validation
pytest>=7.4                # Testing
```

## TDD Implementation Plan

### Phase 1: Embedding Generator (TDD)

#### Test Cases

```python
# tests/test_embedder.py

import pytest
import numpy as np
from src.visualization.embedder import ArticleEmbedder

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
```

#### Implementation

```python
# src/visualization/embedder.py

import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path

class ArticleEmbedder:
    """
    Generate embeddings for research articles.

    Uses sentence transformers to create semantic vectors
    from article titles and abstracts.

    Args:
        model_name: Sentence transformer model name
        device: 'cuda' or 'cpu'

    Example:
        >>> embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')
        >>> embeddings = embedder.embed_articles(articles)
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu'
    ):
        """
        Initialize embedder.

        Recommended models:
        - 'all-MiniLM-L6-v2': Fast, 384-dim (good for testing)
        - 'all-mpnet-base-v2': Better quality, 768-dim
        - 'allenai/specter': Scientific papers, 768-dim
        """
        logger.info(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"✓ Model loaded ({self.embedding_dim}-dim embeddings)")

    def prepare_text(self, article: Dict) -> str:
        """
        Prepare text from article for embedding.

        Combines title and abstract.

        Args:
            article: Article dictionary

        Returns:
            Combined text string
        """
        title = article.get('title', '')
        abstract = article.get('abstract', '')

        # Combine title and abstract
        if abstract:
            text = f"{title}. {abstract}"
        else:
            text = title

        return text.strip()

    def embed_article(
        self,
        article: Dict,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Generate embedding for single article.

        Args:
            article: Article dictionary
            normalize: Normalize to unit length

        Returns:
            Embedding vector
        """
        text = self.prepare_text(article)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        return embedding

    def embed_articles(
        self,
        articles: List[Dict],
        batch_size: int = 32,
        normalize: bool = False,
        show_progress: bool = True,
        cache_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embeddings for multiple articles.

        Args:
            articles: List of article dictionaries
            batch_size: Batch size for encoding
            normalize: Normalize embeddings
            show_progress: Show progress bar
            cache_path: Path to cache embeddings

        Returns:
            Array of embeddings (N, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(articles)} articles...")

        # Prepare texts
        texts = [self.prepare_text(article) for article in articles]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        logger.info(f"✓ Generated {embeddings.shape} embeddings")

        # Cache if requested
        if cache_path:
            self.save_embeddings(embeddings, cache_path)

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, embeddings)
        logger.info(f"✓ Saved embeddings to {path}")

    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings = np.load(path)
        logger.info(f"✓ Loaded embeddings from {path}")
        return embeddings
```

### Phase 2: UMAP Reducer (TDD)

#### Test Cases

```python
# tests/test_reducer.py

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
```

#### Implementation

```python
# src/visualization/reducer.py

import numpy as np
import umap
from typing import Optional
from loguru import logger
import pickle
from pathlib import Path

class UMAPReducer:
    """
    Reduce high-dimensional embeddings to 2D/3D using UMAP.

    UMAP preserves both local and global structure, making it
    ideal for visualization and clustering.

    Args:
        n_components: Output dimensions (2 or 3)
        n_neighbors: Neighbors for local structure (5-50)
        min_dist: Minimum distance between points (0.0-1.0)
        metric: Distance metric ('cosine' or 'euclidean')
        random_state: Random seed for reproducibility

    Example:
        >>> reducer = UMAPReducer(n_components=2)
        >>> reduced = reducer.fit_transform(embeddings)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        # Initialize UMAP
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=False
        )

    def fit(self, embeddings: np.ndarray):
        """
        Fit UMAP on embeddings.

        Args:
            embeddings: High-dimensional embeddings (N, D)
        """
        logger.info(
            f"Fitting UMAP: {embeddings.shape} → {self.n_components}D"
        )

        self.reducer.fit(embeddings)

        logger.info("✓ UMAP fitted")
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted UMAP.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            Reduced embeddings (N, n_components)
        """
        reduced = self.reducer.transform(embeddings)
        return reduced

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP and transform embeddings in one step.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            Reduced embeddings
        """
        logger.info(
            f"Reducing dimensions: {embeddings.shape} → "
            f"({embeddings.shape[0]}, {self.n_components})"
        )

        reduced = self.reducer.fit_transform(embeddings)

        logger.info(f"✓ Reduction complete: {reduced.shape}")
        return reduced

    def save(self, path: str):
        """Save fitted reducer to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"✓ Saved reducer to {path}")

    @staticmethod
    def load(path: str) -> 'UMAPReducer':
        """Load fitted reducer from disk."""
        with open(path, 'rb') as f:
            reducer = pickle.load(f)

        logger.info(f"✓ Loaded reducer from {path}")
        return reducer
```

### Phase 3: HDBSCAN Clusterer (TDD)

#### Test Cases

```python
# tests/test_clusterer.py

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
```

#### Implementation

```python
# src/visualization/clusterer.py

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
            cluster_selection_method='eom'
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
```

### Phase 4: Interactive Plotter (TDD)

#### Test Cases

```python
# tests/test_plotter.py

import pytest
import numpy as np
import pandas as pd
from src.visualization.plotter import InteractivePlotter

@pytest.fixture
def sample_data():
    """Create sample data for plotting."""
    np.random.seed(42)

    return pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'cluster': np.random.randint(0, 5, 100),
        'pmid': [f'{i:08d}' for i in range(100)],
        'title': [f'Paper {i}' for i in range(100)],
        'year': np.random.choice(['2020', '2021', '2022', '2023'], 100)
    })

def test_create_plotter():
    """Should initialize plotter."""
    plotter = InteractivePlotter()
    assert plotter is not None

def test_create_2d_scatter(sample_data):
    """Should create 2D scatter plot."""
    plotter = InteractivePlotter()

    fig = plotter.create_scatter_2d(
        data=sample_data,
        x='x',
        y='y',
        color='cluster',
        hover_data=['pmid', 'title']
    )

    assert fig is not None
    assert len(fig.data) > 0  # Has traces

def test_create_3d_scatter(sample_data):
    """Should create 3D scatter plot."""
    sample_data['z'] = np.random.randn(100)

    plotter = InteractivePlotter()

    fig = plotter.create_scatter_3d(
        data=sample_data,
        x='x',
        y='y',
        z='z',
        color='cluster'
    )

    assert fig is not None

def test_add_cluster_labels(sample_data):
    """Should add cluster labels to plot."""
    plotter = InteractivePlotter()

    cluster_labels = {
        0: 'Immunotherapy',
        1: 'Gene Therapy',
        2: 'Drug Discovery'
    }

    fig = plotter.create_scatter_2d(
        data=sample_data,
        x='x',
        y='y',
        color='cluster',
        cluster_labels=cluster_labels
    )

    # Check if labels appear in legend
    assert fig is not None

def test_save_plot(sample_data, tmp_path):
    """Should save plot to HTML."""
    plotter = InteractivePlotter()

    fig = plotter.create_scatter_2d(
        data=sample_data,
        x='x',
        y='y',
        color='cluster'
    )

    output_path = tmp_path / "plot.html"
    plotter.save_html(fig, str(output_path))

    assert output_path.exists()
```

#### Implementation

```python
# src/visualization/plotter.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger

class InteractivePlotter:
    """
    Create interactive Plotly visualizations.

    Generates scatter plots with hover information,
    filtering, and customization options.

    Example:
        >>> plotter = InteractivePlotter()
        >>> fig = plotter.create_scatter_2d(df, x='x', y='y', color='cluster')
    """

    def __init__(self):
        self.default_colors = px.colors.qualitative.Set3

    def create_scatter_2d(
        self,
        data: pd.DataFrame,
        x: str = 'x',
        y: str = 'y',
        color: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
        cluster_labels: Optional[Dict[int, str]] = None,
        title: str = 'UMAP Projection',
        width: int = 1000,
        height: int = 800
    ) -> go.Figure:
        """
        Create 2D scatter plot.

        Args:
            data: DataFrame with coordinates and metadata
            x: X coordinate column
            y: Y coordinate column
            color: Column to color by
            hover_data: Additional columns to show on hover
            cluster_labels: Mapping of cluster IDs to names
            title: Plot title
            width: Plot width
            height: Plot height

        Returns:
            Plotly Figure
        """
        logger.info(f"Creating 2D scatter plot ({len(data)} points)...")

        # Map cluster labels if provided
        if cluster_labels and color:
            data = data.copy()
            data[f'{color}_label'] = data[color].map(cluster_labels)
            color = f'{color}_label'

        # Create scatter plot
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            hover_data=hover_data,
            title=title,
            width=width,
            height=height
        )

        # Customize
        fig.update_traces(marker=dict(size=5, opacity=0.7))

        fig.update_layout(
            hovermode='closest',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            legend_title=color if color else 'Cluster'
        )

        return fig

    def create_scatter_3d(
        self,
        data: pd.DataFrame,
        x: str = 'x',
        y: str = 'y',
        z: str = 'z',
        color: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
        title: str = 'UMAP 3D Projection'
    ) -> go.Figure:
        """Create 3D scatter plot."""
        logger.info(f"Creating 3D scatter plot ({len(data)} points)...")

        fig = px.scatter_3d(
            data,
            x=x,
            y=y,
            z=z,
            color=color,
            hover_data=hover_data,
            title=title
        )

        fig.update_traces(marker=dict(size=3, opacity=0.6))

        return fig

    def save_html(self, fig: go.Figure, path: str):
        """Save plot as HTML file."""
        fig.write_html(path)
        logger.info(f"✓ Saved plot to {path}")
```

## Running Tests

```bash
# Install dependencies
pip install sentence-transformers umap-learn hdbscan plotly streamlit pandas numpy scikit-learn pytest

# Run all tests
pytest tests/ -v --cov=src/visualization

# Run specific tests
pytest tests/test_embedder.py -v
```

## Usage Examples

### Complete Pipeline

```python
from src.visualization.embedder import ArticleEmbedder
from src.visualization.reducer import UMAPReducer
from src.visualization.clusterer import HDBSCANClusterer
from src.visualization.plotter import InteractivePlotter
import pandas as pd

# 1. Load articles
articles = [...]  # List of article dicts

# 2. Generate embeddings
embedder = ArticleEmbedder(model_name='all-MiniLM-L6-v2')
embeddings = embedder.embed_articles(articles)

# 3. Reduce dimensions
reducer = UMAPReducer(n_components=2)
reduced = reducer.fit_transform(embeddings)

# 4. Cluster
clusterer = HDBSCANClusterer(min_cluster_size=50)
labels = clusterer.fit_predict(reduced)

# 5. Visualize
df = pd.DataFrame({
    'x': reduced[:, 0],
    'y': reduced[:, 1],
    'cluster': labels,
    'pmid': [a['pmid'] for a in articles],
    'title': [a['title'] for a in articles],
    'year': [a['year'] for a in articles]
})

plotter = InteractivePlotter()
fig = plotter.create_scatter_2d(
    data=df,
    color='cluster',
    hover_data=['pmid', 'title', 'year']
)

fig.show()
```

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Generate embeddings for articles
3. UMAP dimensionality reduction (2D and 3D)
4. HDBSCAN clustering
5. Interactive Plotly scatter plots
6. Hover information displays article details

✅ **Should Have:**
7. Cluster labeling
8. Export to HTML
9. Filter by year/cluster
10. Search functionality

✅ **Nice to Have:**
11. Streamlit dashboard
12. Real-time updates
13. Cluster naming with LLM

## Deliverables

1. **Source code** (>90% coverage)
2. **Tests**
3. **Sample visualizations** (HTML files)
4. **Documentation**
5. **Example notebooks**

## Environment Setup

```bash
# Install dependencies
pip install sentence-transformers umap-learn hdbscan plotly streamlit pandas numpy scikit-learn pytest

# Run tests
pytest tests/ -v --cov=src/visualization
```

---

**Task completion**: When all tests pass, embeddings generate correctly, UMAP reduces dimensions, HDBSCAN clusters, and interactive plots display properly.
