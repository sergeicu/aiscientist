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
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {int(k) if isinstance(k, np.integer) else k: convert_to_native(v)
                        for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        summary = {
            'labels': convert_to_native(cluster_labels),
            'statistics': convert_to_native(stats)
        }

        if keywords:
            summary['keywords'] = convert_to_native(keywords)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Exported summary to {output_path}")
