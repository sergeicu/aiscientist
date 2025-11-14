"""Embedding generation and vector store for semantic search."""

from .embedder import ArticleEmbedder
from .vector_store import VectorStore

__all__ = ['ArticleEmbedder', 'VectorStore']
