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
