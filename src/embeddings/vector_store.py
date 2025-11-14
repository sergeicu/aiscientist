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
