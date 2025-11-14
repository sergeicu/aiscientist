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
