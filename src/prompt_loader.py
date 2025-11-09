"""
Prompt template loader and formatter.

Loads YAML prompt templates and formats them with article data.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from loguru import logger

from .models import Article


class PromptLoader:
    """Loads and formats prompt templates."""

    def __init__(self, template_path: Path):
        """
        Initialize prompt loader.

        Args:
            template_path: Path to YAML template file
        """
        self.template_path = template_path
        self.template_data: Optional[Dict[str, Any]] = None
        self._load_template()

    def _load_template(self) -> None:
        """Load template from YAML file."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {self.template_path}")

        logger.info(f"Loading prompt template: {self.template_path}")

        with open(self.template_path, 'r', encoding='utf-8') as f:
            self.template_data = yaml.safe_load(f)

        logger.info("✓ Template loaded successfully")

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        if not self.template_data:
            raise RuntimeError("Template not loaded")

        return self.template_data.get('system_prompt', '').strip()

    def format_prompt(
        self,
        article: Article,
        include_examples: bool = True,
        num_examples: int = 2
    ) -> str:
        """
        Format prompt for an article.

        Args:
            article: Article to classify
            include_examples: Whether to include few-shot examples
            num_examples: Number of examples to include

        Returns:
            Formatted prompt string
        """
        if not self.template_data:
            raise RuntimeError("Template not loaded")

        # Build prompt sections
        sections = []

        # Task description
        task_desc = self.template_data.get('task_description', '').strip()
        if task_desc:
            sections.append(task_desc)

        # Output format instructions
        format_instructions = self.template_data.get('output_format_instructions', '').strip()
        if format_instructions:
            sections.append(format_instructions)

        # Few-shot examples
        if include_examples:
            examples_section = self._format_examples(num_examples)
            if examples_section:
                sections.append(examples_section)

        # Edge case guidance (optional, can be included for harder cases)
        # edge_guidance = self.template_data.get('edge_case_guidance', '').strip()
        # if edge_guidance:
        #     sections.append(edge_guidance)

        # Current article
        article_section = self._format_article_section(article)
        sections.append(article_section)

        # Final reminder
        sections.append("Remember: Return ONLY valid JSON matching the schema, no additional text.")

        return "\n\n".join(sections)

    def _format_examples(self, num_examples: int) -> str:
        """Format few-shot examples."""
        examples = self.template_data.get('examples', [])

        if not examples:
            return ""

        # Limit number of examples
        examples = examples[:num_examples]

        lines = ["Here are some examples to guide you:"]
        lines.append("")

        for i, example in enumerate(examples, 1):
            label = example.get('label', f'Example {i}')
            abstract = example.get('abstract', '').strip()
            expected = example.get('expected_output', {})

            lines.append(f"--- Example {i}: {label} ---")
            lines.append(f"Abstract: {abstract}")
            lines.append("")
            lines.append("Expected Output:")

            # Format expected output as JSON
            import json
            lines.append(json.dumps(expected, indent=2))
            lines.append("")

        return "\n".join(lines)

    def _format_article_section(self, article: Article) -> str:
        """Format the article section."""
        lines = ["--- Article to Classify ---"]

        if article.title:
            lines.append(f"Title: {article.title}")
            lines.append("")

        lines.append(f"Abstract: {article.abstract}")
        lines.append("")
        lines.append("Your Classification (JSON only):")

        return "\n".join(lines)

    def get_raw_template(self) -> Dict[str, Any]:
        """Get raw template data."""
        if not self.template_data:
            raise RuntimeError("Template not loaded")
        return self.template_data
