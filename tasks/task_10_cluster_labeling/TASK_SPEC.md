# Task 10: Cluster Labeling & Topic Analysis - TDD Implementation

## Executive Summary

Implement automatic cluster labeling using TF-IDF keywords and optional LLM-based naming. This makes clusters interpretable by extracting representative terms and generating human-readable labels.

**Key Requirements:**
- Extract representative keywords using TF-IDF
- Generate cluster labels from keywords
- Calculate cluster statistics
- Optional: Use LLM for better naming
- Export cluster summaries
- Follow TDD principles

**Dependencies**: Requires Task 9 (cluster assignments) to provide input.

## Why Automatic Labeling?

**Problem**: Clusters are numbered (0, 1, 2...) - not interpretable
**Solution**: Extract representative terms and generate meaningful labels

**Methods**:
1. **TF-IDF**: Statistical keyword extraction (fast, no API costs)
2. **c-TF-IDF**: Class-based TF-IDF (better for clusters)
3. **LLM**: GPT-4 generates descriptive names (best quality, costs money)

## TDD Implementation

### Test Cases

```python
# tests/test_cluster_labeling.py

import pytest
import pandas as pd
from src.labeling.labeler import ClusterLabeler

@pytest.fixture
def sample_data():
    """Create sample clustered articles."""
    return pd.DataFrame({
        'pmid': ['1', '2', '3', '4', '5', '6'],
        'title': [
            'CAR-T therapy for leukemia',
            'CAR-T cells in lymphoma',
            'CRISPR gene editing sickle cell',
            'CRISPR Cas9 therapeutic applications',
            'Checkpoint inhibitors melanoma',
            'PD-1 blockade cancer treatment'
        ],
        'abstract': [
            'CAR-T cell therapy shows promise...',
            'Chimeric antigen receptor T cells...',
            'Gene editing using CRISPR-Cas9...',
            'CRISPR therapeutic potential...',
            'Immune checkpoint inhibitors...',
            'PD-1 and PD-L1 blockade...'
        ],
        'cluster': [0, 0, 1, 1, 2, 2]
    })

def test_create_labeler():
    """Should initialize labeler."""
    labeler = ClusterLabeler()
    assert labeler is not None

def test_extract_keywords_tfidf(sample_data):
    """Should extract keywords using TF-IDF."""
    labeler = ClusterLabeler()
    keywords = labeler.extract_keywords_tfidf(
        texts=sample_data['title'] + ' ' + sample_data['abstract'],
        labels=sample_data['cluster'],
        n_keywords=5
    )

    # Should have keywords for each cluster
    assert 0 in keywords
    assert 1 in keywords
    assert 2 in keywords

    # Cluster 0 should have CAR-T keywords
    assert any('car' in k.lower() or 't cell' in k.lower()
               for k in keywords[0])

def test_generate_label_from_keywords(sample_data):
    """Should generate label from keywords."""
    labeler = ClusterLabeler()

    keywords = ['CAR-T', 'therapy', 'leukemia', 'lymphoma', 'cells']
    label = labeler.generate_label(keywords)

    assert isinstance(label, str)
    assert len(label) > 0
    assert 'CAR-T' in label or 'car' in label.lower()

def test_label_all_clusters(sample_data):
    """Should label all clusters."""
    labeler = ClusterLabeler()
    labels = labeler.label_clusters(
        texts=sample_data['title'] + ' ' + sample_data['abstract'],
        cluster_labels=sample_data['cluster']
    )

    # Should have label for each cluster
    assert len(labels) == 3
    assert all(isinstance(label, str) for label in labels.values())

def test_get_cluster_statistics(sample_data):
    """Should calculate cluster statistics."""
    labeler = ClusterLabeler()

    stats = labeler.get_cluster_stats(
        data=sample_data,
        cluster_col='cluster'
    )

    assert 'cluster_sizes' in stats
    assert stats['cluster_sizes'][0] == 2  # 2 articles in cluster 0
    assert stats['total_clusters'] == 3

def test_get_representative_docs(sample_data):
    """Should find most representative documents."""
    import numpy as np

    # Create fake embeddings
    embeddings = np.random.randn(6, 10)

    labeler = ClusterLabeler()
    reps = labeler.get_representative_docs(
        cluster_id=0,
        cluster_labels=sample_data['cluster'],
        embeddings=embeddings,
        doc_ids=sample_data['pmid'].tolist(),
        n_docs=1
    )

    assert len(reps) == 1
    assert reps[0] in ['1', '2']  # Should be from cluster 0

def test_export_cluster_summary(sample_data, tmp_path):
    """Should export cluster summary."""
    labeler = ClusterLabeler()

    labels = labeler.label_clusters(
        texts=sample_data['title'] + ' ' + sample_data['abstract'],
        cluster_labels=sample_data['cluster']
    )

    output_path = tmp_path / "clusters.json"
    labeler.export_summary(
        cluster_labels=labels,
        stats=labeler.get_cluster_stats(sample_data, 'cluster'),
        output_path=str(output_path)
    )

    assert output_path.exists()
```

### Implementation

```python
# src/labeling/labeler.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
from loguru import logger
import json

class ClusterLabeler:
    """
    Generate interpretable labels for clusters.

    Uses TF-IDF for keyword extraction and simple heuristics
    for label generation.

    Example:
        >>> labeler = ClusterLabeler()
        >>> labels = labeler.label_clusters(texts, cluster_labels)
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features

    def extract_keywords_tfidf(
        self,
        texts: pd.Series,
        labels: pd.Series,
        n_keywords: int = 10
    ) -> Dict[int, List[str]]:
        """
        Extract top keywords for each cluster using TF-IDF.

        Class-based TF-IDF: compares cluster vs. all other clusters.
        """
        logger.info("Extracting keywords with TF-IDF...")

        # Concatenate documents by cluster
        cluster_texts = {}
        for cluster_id in labels.unique():
            if cluster_id == -1:  # Skip noise
                continue
            mask = labels == cluster_id
            cluster_texts[cluster_id] = ' '.join(texts[mask].tolist())

        # Fit TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(cluster_texts.values())
        feature_names = vectorizer.get_feature_names_out()

        # Extract top keywords for each cluster
        keywords = {}
        for idx, cluster_id in enumerate(cluster_texts.keys()):
            # Get TF-IDF scores for this cluster
            scores = tfidf_matrix[idx].toarray().flatten()

            # Get top N keywords
            top_indices = scores.argsort()[-n_keywords:][::-1]
            keywords[cluster_id] = [feature_names[i] for i in top_indices]

        logger.info(f"✓ Extracted keywords for {len(keywords)} clusters")
        return keywords

    def generate_label(self, keywords: List[str], max_words: int = 4) -> str:
        """
        Generate cluster label from keywords.

        Simple heuristic: take top keywords and capitalize.
        """
        # Take first few keywords
        label_words = keywords[:max_words]

        # Capitalize and join
        label = ' & '.join(word.title() for word in label_words)

        return label

    def label_clusters(
        self,
        texts: pd.Series,
        cluster_labels: pd.Series,
        n_keywords: int = 10
    ) -> Dict[int, str]:
        """
        Generate labels for all clusters.

        Returns:
            Dictionary mapping cluster_id to label string
        """
        logger.info("Generating cluster labels...")

        # Extract keywords
        keywords = self.extract_keywords_tfidf(texts, cluster_labels, n_keywords)

        # Generate labels
        labels = {}
        for cluster_id, kws in keywords.items():
            labels[cluster_id] = self.generate_label(kws)

        # Noise cluster
        if -1 in cluster_labels.unique():
            labels[-1] = "Outliers/Noise"

        logger.info(f"✓ Generated {len(labels)} labels")
        return labels

    def get_cluster_stats(
        self,
        data: pd.DataFrame,
        cluster_col: str = 'cluster'
    ) -> Dict:
        """Calculate cluster statistics."""
        stats = {
            'total_clusters': len(data[cluster_col].unique()),
            'cluster_sizes': data[cluster_col].value_counts().to_dict(),
            'total_docs': len(data)
        }

        # Calculate noise percentage
        if -1 in stats['cluster_sizes']:
            stats['noise_percentage'] = stats['cluster_sizes'][-1] / len(data) * 100
        else:
            stats['noise_percentage'] = 0

        return stats

    def get_representative_docs(
        self,
        cluster_id: int,
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        doc_ids: List[str],
        n_docs: int = 5
    ) -> List[str]:
        """
        Find most representative documents in cluster.

        Returns documents closest to cluster centroid.
        """
        # Get cluster mask
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            return []

        cluster_embeddings = embeddings[mask]
        cluster_doc_ids = [doc_ids[i] for i, m in enumerate(mask) if m]

        # Calculate centroid
        centroid = cluster_embeddings.mean(axis=0)

        # Find closest documents to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_indices = distances.argsort()[:n_docs]

        return [cluster_doc_ids[i] for i in closest_indices]

    def export_summary(
        self,
        cluster_labels: Dict[int, str],
        stats: Dict,
        output_path: str,
        keywords: Dict[int, List[str]] = None
    ):
        """Export cluster summary to JSON."""
        summary = {
            'labels': cluster_labels,
            'statistics': stats
        }

        if keywords:
            summary['keywords'] = keywords

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Exported summary to {output_path}")
```

### Optional: LLM-Based Labeling

```python
# src/labeling/llm_labeler.py (optional, requires API key)

import openai
from typing import List

class LLMLabeler:
    """
    Use LLM to generate cluster labels.

    Example:
        >>> labeler = LLMLabeler(api_key="sk-...")
        >>> label = labeler.generate_label(sample_titles, keywords)
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview"):
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate_label(
        self,
        sample_titles: List[str],
        keywords: List[str],
        max_length: int = 50
    ) -> str:
        """Generate descriptive label using LLM."""

        prompt = f"""
        Analyze this cluster of research articles and generate a concise,
        descriptive label (max {max_length} characters).

        Sample titles:
        {chr(10).join(f"- {t}" for t in sample_titles[:5])}

        Top keywords:
        {', '.join(keywords[:10])}

        Generate a clear, specific label that describes the research theme.
        Label:
        """

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )

        label = response.choices[0].message.content.strip()
        return label
```

## Usage Example

```python
from src.labeling.labeler import ClusterLabeler
import pandas as pd

# Load clustered data
df = pd.read_csv('cluster_assignments.csv')

# Load articles for text
with open('articles.json') as f:
    articles = json.load(f)

df['text'] = [f"{a['title']}. {a['abstract']}" for a in articles]

# Label clusters
labeler = ClusterLabeler()
labels = labeler.label_clusters(
    texts=df['text'],
    cluster_labels=df['cluster']
)

# Print labels
for cluster_id, label in labels.items():
    size = (df['cluster'] == cluster_id).sum()
    print(f"Cluster {cluster_id} ({size} articles): {label}")

# Export summary
stats = labeler.get_cluster_stats(df, 'cluster')
labeler.export_summary(labels, stats, 'cluster_summary.json')
```

## Success Criteria

✅ Tests pass (>90% coverage)
✅ TF-IDF extracts relevant keywords
✅ Labels are interpretable
✅ Statistics calculated correctly
✅ Summary exports successfully

## Running Tests

```bash
pip install scikit-learn pandas pytest

pytest tests/test_cluster_labeling.py -v
```

---

**Task completion**: When all tests pass and generated labels are meaningful and interpretable.
