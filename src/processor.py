"""
Main processor for classifying PubMed articles.

Orchestrates all components:
- CSV parsing
- Prompt formatting
- Ollama generation
- Structured output parsing
- Checkpoint management
- Results saving
"""

import json
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from loguru import logger

from .config import Config
from .models import (
    Article,
    ProcessingResult,
    ProcessingCheckpoint,
    ProcessingStats,
    ClassificationResult
)
from .csv_parser import CSVParser
from .ollama_client import OllamaClient
from .structured_output import StructuredOutputHandler
from .prompt_loader import PromptLoader


class ArticleProcessor:
    """Main processor for article classification."""

    def __init__(self, config: Config):
        """
        Initialize processor.

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize components
        self.csv_parser = CSVParser(config)
        self.ollama_client = OllamaClient(config)
        self.output_handler = StructuredOutputHandler(config)
        self.prompt_loader = PromptLoader(config.prompt_template)

        # State
        self.checkpoint: Optional[ProcessingCheckpoint] = None
        self.stats = ProcessingStats(
            total_articles=0,
            processed=0,
            successful=0,
            failed=0
        )
        self.output_file: Optional[Path] = None
        self.results: List[ProcessingResult] = []

    def initialize(self) -> bool:
        """
        Initialize processor and check all components.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing article processor...")

        # Load CSV
        try:
            self.csv_parser.load_and_validate()
            self.stats.total_articles = self.csv_parser.get_total_count()
            logger.info(f"✓ CSV loaded: {self.stats.total_articles} articles")
        except Exception as e:
            logger.error(f"✗ CSV loading failed: {e}")
            return False

        # Check Ollama connection
        if not self.ollama_client.check_connection():
            return False

        # Check model availability
        if not self.ollama_client.ensure_model_ready():
            return False

        # Warm up model (optional but recommended)
        logger.info("Warming up model...")
        if not self.ollama_client.warmup():
            logger.warning("Model warmup failed, continuing anyway...")

        # Load or create checkpoint
        if self.config.resume_from_checkpoint:
            self._load_checkpoint()

        # Create output file
        self.output_file = self.config.get_output_filename("classification_results")
        logger.info(f"Output file: {self.output_file}")

        logger.info("✓ Initialization complete")
        return True

    def process_all(self) -> ProcessingStats:
        """
        Process all articles in the CSV.

        Returns:
            Final processing statistics
        """
        logger.info("Starting article processing...")

        start_index = 0
        if self.checkpoint:
            start_index = self.checkpoint.last_processed_index + 1
            logger.info(f"Resuming from article {start_index}")

        total = self.csv_parser.get_total_count()
        system_prompt = self.prompt_loader.get_system_prompt()

        # Process articles
        for idx, article in self.csv_parser.iterate_articles(start_index=start_index):
            logger.info(f"Processing article {idx + 1}/{total} (PMID: {article.pmid})")

            result = self._process_single_article(article, system_prompt)
            self.results.append(result)

            # Update statistics
            self.stats.update_from_result(result)

            # Save result immediately (JSONL format)
            self._save_result(result)

            # Save checkpoint periodically
            if (idx + 1) % self.config.batch_size == 0:
                self._save_checkpoint(idx)
                self._log_progress()

        # Final checkpoint
        self._save_checkpoint(total - 1)
        self._log_final_stats()

        return self.stats

    def process_batch(
        self,
        start_index: int,
        batch_size: int
    ) -> List[ProcessingResult]:
        """
        Process a specific batch of articles.

        Args:
            start_index: Starting index
            batch_size: Number of articles to process

        Returns:
            List of processing results
        """
        logger.info(f"Processing batch: {start_index} to {start_index + batch_size}")

        system_prompt = self.prompt_loader.get_system_prompt()
        batch_results = []

        batch_articles = self.csv_parser.get_articles_batch(start_index, batch_size)

        for idx, article in batch_articles:
            result = self._process_single_article(article, system_prompt)
            batch_results.append(result)
            self.stats.update_from_result(result)

        return batch_results

    def _process_single_article(
        self,
        article: Article,
        system_prompt: str
    ) -> ProcessingResult:
        """
        Process a single article.

        Args:
            article: Article to classify
            system_prompt: System prompt for LLM

        Returns:
            Processing result
        """
        start_time = time.time()

        result = ProcessingResult(
            article=article,
            model_used=self.config.ollama_model
        )

        try:
            # Format prompt
            user_prompt = self.prompt_loader.format_prompt(
                article,
                include_examples=True,
                num_examples=2
            )

            # Generate with Ollama
            response = self.ollama_client.generate_with_retry(
                prompt=user_prompt,
                system_prompt=system_prompt,
                format='json'  # Request JSON format from Ollama
            )

            if not response:
                raise Exception("No response from Ollama after retries")

            # Extract response content
            raw_output = response.get('message', {}).get('content', '')
            result.raw_output = raw_output

            # Parse structured output
            classification, parsing_method = self.output_handler.parse_json_output(raw_output)

            if classification:
                # Validate
                if self.output_handler.validate_result(classification):
                    result.classification = classification
                    result.parsing_method = parsing_method
                    result.success = True
                    logger.info(
                        f"✓ Classification: {'Trial' if classification.is_clinical_trial else 'Not Trial'} "
                        f"(confidence: {classification.confidence:.2f}, method: {parsing_method})"
                    )
                else:
                    result.error = "Validation failed"
                    logger.warning("Validation failed for classification result")
            else:
                result.error = f"Failed to parse output (tried all methods)"
                logger.error("Failed to parse structured output")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Processing error: {e}")

        # Record timing
        result.processing_time_seconds = time.time() - start_time

        return result

    def _save_result(self, result: ProcessingResult) -> None:
        """Save a single result to output file (JSONL format)."""
        if not self.output_file:
            return

        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                # Convert to dict and write as JSON line
                json_line = result.model_dump_json()
                f.write(json_line + '\n')
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def _save_checkpoint(self, last_index: int) -> None:
        """Save processing checkpoint."""
        checkpoint = ProcessingCheckpoint(
            last_processed_index=last_index,
            total_processed=self.stats.processed,
            successful=self.stats.successful,
            failed=self.stats.failed,
            output_file=str(self.output_file) if self.output_file else ""
        )

        checkpoint_file = self.config.get_checkpoint_filename()

        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                f.write(checkpoint.model_dump_json(indent=2))
            logger.debug(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> None:
        """Load existing checkpoint if available."""
        checkpoint_file = self.config.get_checkpoint_filename()

        if not checkpoint_file.exists():
            logger.info("No checkpoint found, starting from beginning")
            return

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.checkpoint = ProcessingCheckpoint(**data)

            logger.info(
                f"✓ Loaded checkpoint: {self.checkpoint.total_processed} articles processed, "
                f"last index: {self.checkpoint.last_processed_index}"
            )

            # Restore output file path
            if self.checkpoint.output_file:
                self.output_file = Path(self.checkpoint.output_file)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            self.checkpoint = None

    def _log_progress(self) -> None:
        """Log current progress."""
        total = self.stats.total_articles
        processed = self.stats.processed
        percent = (processed / total * 100) if total > 0 else 0

        logger.info("=" * 60)
        logger.info(f"Progress: {processed}/{total} ({percent:.1f}%)")
        logger.info(f"Successful: {self.stats.successful}")
        logger.info(f"Failed: {self.stats.failed}")
        logger.info(f"Clinical Trials: {self.stats.clinical_trials}")
        logger.info(f"Non-Trials: {self.stats.non_trials}")

        if self.stats.average_time_per_article > 0:
            remaining = total - processed
            eta_seconds = remaining * self.stats.average_time_per_article
            eta_minutes = eta_seconds / 60
            logger.info(f"Avg time/article: {self.stats.average_time_per_article:.2f}s")
            logger.info(f"ETA: {eta_minutes:.1f} minutes")

        logger.info("=" * 60)

    def _log_final_stats(self) -> None:
        """Log final statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total articles: {self.stats.total_articles}")
        logger.info(f"Processed: {self.stats.processed}")
        logger.info(f"Successful: {self.stats.successful}")
        logger.info(f"Failed: {self.stats.failed}")
        logger.info("")
        logger.info(f"Clinical Trials: {self.stats.clinical_trials}")
        logger.info(f"Non-Trials: {self.stats.non_trials}")
        logger.info("")
        logger.info(f"Total time: {self.stats.total_time_seconds:.1f}s")
        logger.info(f"Average time/article: {self.stats.average_time_per_article:.2f}s")
        logger.info("")

        if self.stats.parsing_methods:
            logger.info("Parsing methods used:")
            for method, count in self.stats.parsing_methods.items():
                logger.info(f"  {method}: {count}")
            logger.info("")

        if self.stats.trial_phases:
            logger.info("Trial phases:")
            for phase, count in self.stats.trial_phases.items():
                logger.info(f"  {phase}: {count}")
            logger.info("")

        if self.output_file:
            logger.info(f"Results saved to: {self.output_file}")

        logger.info("=" * 60)

        # Save final stats to JSON
        stats_file = self.config.output_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(self.stats.model_dump_json(indent=2))
            logger.info(f"Statistics saved to: {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
