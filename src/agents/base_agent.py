"""Base agent class for multi-agent system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from loguru import logger
import yaml
from pathlib import Path


class AgentMessage(BaseModel):
    """Inter-agent communication message."""

    sender: str = Field(..., description="Agent name sending the message")
    recipient: str = Field(..., description="Agent name receiving the message")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""

    def __init__(
        self,
        name: str,
        llm_client,  # OllamaClient
        memory_manager,  # MemoryManager
        prompt_dir: Path = None
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier
            llm_client: LLM client for generation
            memory_manager: Shared memory system
            prompt_dir: Directory containing prompt templates
        """
        self.name = name
        self.llm = llm_client
        self.memory = memory_manager
        self.prompt_dir = prompt_dir or Path("prompts/agents")
        self.prompts = self._load_prompts()

        logger.info(f"Initialized {self.name} agent")

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompt templates from YAML file."""
        prompt_file = self.prompt_dir / f"{self.name}.yaml"

        if not prompt_file.exists():
            logger.warning(f"Prompt file not found: {prompt_file}")
            return {}

        with open(prompt_file) as f:
            prompts = yaml.safe_load(f)

        logger.debug(f"Loaded prompts for {self.name}")
        return prompts

    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Get and format a prompt template.

        Args:
            template_name: Name of template in YAML file
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt string
        """
        if template_name not in self.prompts:
            raise ValueError(f"Template '{template_name}' not found for {self.name}")

        template = self.prompts[template_name]

        # Format with provided kwargs
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable {e} for template '{template_name}'")

    @abstractmethod
    async def process(self, task: Dict) -> Dict:
        """
        Process a task assigned to this agent.

        Args:
            task: Task specification with type and parameters

        Returns:
            Task result dictionary
        """
        pass

    async def query_memory(
        self,
        query: str,
        collection: str = "articles",
        n_results: int = 10
    ) -> List[Dict]:
        """
        Query shared memory with semantic search.

        Args:
            query: Search query
            collection: Memory collection to search
            n_results: Number of results

        Returns:
            List of matching documents with metadata
        """
        return await self.memory.semantic_search(
            query=query,
            collection=collection,
            n_results=n_results
        )

    async def store_finding(
        self,
        finding: str,
        metadata: Dict = None
    ):
        """
        Store a finding in long-term memory.

        Args:
            finding: Finding text to store
            metadata: Additional metadata
        """
        await self.memory.store_finding(
            agent_name=self.name,
            finding=finding,
            metadata=metadata or {}
        )

    async def send_message(
        self,
        recipient: str,
        content: str,
        metadata: Dict = None
    ) -> AgentMessage:
        """
        Send message to another agent.

        Args:
            recipient: Target agent name
            content: Message content
            metadata: Additional metadata

        Returns:
            Created message
        """
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            metadata=metadata or {}
        )

        logger.debug(f"{self.name} → {recipient}: {content[:100]}...")
        return message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
