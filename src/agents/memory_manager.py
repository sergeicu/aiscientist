"""Shared memory system for multi-agent collaboration."""

from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from loguru import logger


class MemoryManager:
    """Manage shared memory across all agents."""

    def __init__(
        self,
        chroma_path: str = "./data/chroma",
        persist: bool = True
    ):
        """
        Initialize memory manager.

        Args:
            chroma_path: Path to ChromaDB storage
            persist: Whether to persist data
        """
        self.chroma_path = chroma_path

        if persist:
            self.client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client()

        # Initialize collections
        self.articles = self._get_or_create_collection(
            "articles",
            "PubMed article embeddings"
        )
        self.findings = self._get_or_create_collection(
            "findings",
            "Agent findings and insights"
        )
        self.conversations = self._get_or_create_collection(
            "conversations",
            "Agent conversation history"
        )

        logger.info(f"Memory manager initialized with {self.articles.count()} articles")

    def _get_or_create_collection(
        self,
        name: str,
        description: str
    ):
        """Get or create a ChromaDB collection."""
        try:
            collection = self.client.get_collection(name)
            logger.debug(f"Loaded existing collection: {name}")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata={"description": description}
            )
            logger.debug(f"Created new collection: {name}")

        return collection

    async def semantic_search(
        self,
        query: str,
        collection: str = "articles",
        n_results: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform semantic search across a collection.

        Args:
            query: Search query
            collection: Collection name (articles, findings, conversations)
            n_results: Number of results to return
            filter: Metadata filter (ChromaDB where clause)

        Returns:
            List of results with documents and metadata
        """
        coll = getattr(self, collection)

        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where=filter
        )

        # Format results
        formatted = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "id": results["ids"][0][i] if results.get("ids") else None
                })

        logger.debug(f"Semantic search: '{query[:50]}...' → {len(formatted)} results")
        return formatted

    async def store_finding(
        self,
        agent_name: str,
        finding: str,
        metadata: Dict
    ):
        """
        Store an agent finding in long-term memory.

        Args:
            agent_name: Name of agent storing finding
            finding: Finding text
            metadata: Additional metadata
        """
        finding_id = f"{agent_name}_{datetime.now().timestamp()}"

        self.findings.add(
            documents=[finding],
            metadatas=[{
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }],
            ids=[finding_id]
        )

        logger.debug(f"Stored finding from {agent_name}: {finding[:50]}...")

    async def store_conversation(
        self,
        conversation_id: str,
        content: str,
        metadata: Dict
    ):
        """
        Store conversation history.

        Args:
            conversation_id: Unique conversation identifier
            content: Conversation content
            metadata: Metadata (participants, topic, etc.)
        """
        self.conversations.add(
            documents=[content],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                **metadata
            }],
            ids=[conversation_id]
        )

        logger.debug(f"Stored conversation: {conversation_id}")

    async def retrieve_context(
        self,
        query: str,
        max_articles: int = 5,
        max_findings: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve relevant context for a query.

        Searches across articles and findings to provide context.

        Args:
            query: Context query
            max_articles: Max articles to retrieve
            max_findings: Max findings to retrieve

        Returns:
            Dictionary with articles and findings
        """
        # Search articles
        article_results = await self.semantic_search(
            query=query,
            collection="articles",
            n_results=max_articles
        )

        # Search findings
        finding_results = await self.semantic_search(
            query=query,
            collection="findings",
            n_results=max_findings
        )

        logger.info(
            f"Retrieved context: {len(article_results)} articles, "
            f"{len(finding_results)} findings"
        )

        return {
            "articles": article_results,
            "findings": finding_results
        }

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections."""
        return {
            "articles": self.articles.count(),
            "findings": self.findings.count(),
            "conversations": self.conversations.count()
        }

    def clear_collection(self, collection_name: str):
        """Clear a collection (use with caution!)."""
        coll = getattr(self, collection_name)
        self.client.delete_collection(collection_name)

        # Recreate
        setattr(
            self,
            collection_name,
            self._get_or_create_collection(
                collection_name,
                f"Cleared {collection_name}"
            )
        )

        logger.warning(f"Cleared collection: {collection_name}")
