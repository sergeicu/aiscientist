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
