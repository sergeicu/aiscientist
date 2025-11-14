"""
Ollama client for interacting with local language models.

Provides:
- Connection health checks
- Prompt formatting
- Streaming and non-streaming generation
- Error handling and retries
- Rate limiting
"""

import time
from typing import Optional, Dict, Any
import ollama
from loguru import logger

from .config import Config


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, config: Config):
        """
        Initialize Ollama client.

        Args:
            config: Application configuration
        """
        self.config = config
        self.client = ollama.Client(host=config.ollama_base_url)
        self.model_name = config.ollama_model

        # Rate limiting
        self.last_request_time: float = 0.0
        self.min_request_interval: float = 0.1  # 100ms between requests

    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to list models
            self.client.list()
            logger.info("✓ Ollama connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Ollama connection failed: {e}")
            return False

    def check_model_availability(self) -> bool:
        """
        Check if the specified model is available.

        Returns:
            True if model is available, False otherwise
        """
        try:
            models = self.client.list()
            model_names = [model['name'] for model in models.get('models', [])]

            # Check for exact match or partial match
            available = any(
                self.model_name in name or name in self.model_name
                for name in model_names
            )

            if available:
                logger.info(f"✓ Model '{self.model_name}' is available")
                return True
            else:
                logger.warning(f"✗ Model '{self.model_name}' not found")
                logger.info(f"Available models: {model_names}")
                return False

        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    def pull_model(self) -> bool:
        """
        Pull the model if not available.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model '{self.model_name}'...")
            self.client.pull(self.model_name)
            logger.info(f"✓ Model '{self.model_name}' pulled successfully")
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False

    def ensure_model_ready(self) -> bool:
        """
        Ensure model is available, pulling if necessary.

        Returns:
            True if model is ready, False otherwise
        """
        if self.check_model_availability():
            return True

        logger.info(f"Model '{self.model_name}' not found, attempting to pull...")
        return self.pull_model()

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate completion from the model.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            format: Response format (e.g., 'json')
            options: Additional options for generation

        Returns:
            Response dictionary from Ollama

        Raises:
            Exception: If generation fails
        """
        self._rate_limit()

        # Build request
        request_options = {
            'num_ctx': self.config.ollama_num_ctx,
            'temperature': self.config.ollama_temperature,
        }

        if options:
            request_options.update(options)

        try:
            # Build messages
            messages = []

            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })

            messages.append({
                'role': 'user',
                'content': prompt
            })

            # Generate
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format=format,
                options=request_options,
                stream=False
            )

            return response

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate with automatic retry on failure.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            format: Response format (e.g., 'json')
            options: Additional options for generation
            max_retries: Maximum retry attempts (uses config if not specified)
            retry_delay: Delay between retries in seconds (uses config if not specified)

        Returns:
            Response dictionary or None if all retries fail
        """
        max_retries = max_retries or self.config.max_retries
        retry_delay = retry_delay or self.config.retry_delay_seconds

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    format=format,
                    options=options
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Generation attempt {attempt + 1}/{max_retries + 1} failed: {e}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"All {max_retries + 1} generation attempts failed. Last error: {e}"
                    )

        return None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current model.

        Returns:
            Model information dictionary or None if unavailable
        """
        try:
            return self.client.show(self.model_name)
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None

    def test_generation(self) -> bool:
        """
        Test generation with a simple prompt.

        Returns:
            True if test successful, False otherwise
        """
        try:
            logger.info("Testing generation with simple prompt...")

            response = self.generate(
                prompt="Say 'OK' if you can read this.",
                system_prompt="You are a helpful assistant. Respond concisely."
            )

            if response and 'message' in response:
                content = response['message'].get('content', '')
                logger.info(f"✓ Test generation successful: {content[:100]}")
                return True
            else:
                logger.error("✗ Test generation failed: No response")
                return False

        except Exception as e:
            logger.error(f"✗ Test generation failed: {e}")
            return False

    def warmup(self) -> bool:
        """
        Warm up the model with a test generation.

        This can help load the model into memory before actual processing.

        Returns:
            True if warmup successful, False otherwise
        """
        logger.info("Warming up model...")
        return self.test_generation()

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output matching a schema.

        Args:
            prompt: User prompt
            schema: Expected output schema (simple format)
            system_prompt: System prompt (optional)
            max_retries: Maximum retry attempts

        Returns:
            Parsed JSON dictionary matching schema

        Raises:
            Exception: If generation or parsing fails after retries
        """
        import json
        from json_repair import repair_json

        # Add schema instruction to prompt
        schema_str = json.dumps(schema, indent=2)
        structured_prompt = f"{prompt}\n\nProvide response as valid JSON matching this schema:\n{schema_str}"

        last_error = None

        for attempt in range(max_retries):
            try:
                # Generate with JSON format
                response = self.generate(
                    prompt=structured_prompt,
                    system_prompt=system_prompt,
                    format="json"
                )

                # Extract content
                content = response.get("message", {}).get("content", "")

                # Try parsing directly
                try:
                    result = json.loads(content)
                    logger.debug(f"Structured generation successful on attempt {attempt + 1}")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode failed, attempting repair: {e}")
                    # Try repairing JSON
                    repaired = repair_json(content)
                    result = json.loads(repaired)
                    logger.debug(f"JSON repaired and parsed successfully")
                    return result

            except Exception as e:
                last_error = e
                logger.warning(f"Structured generation attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                    time.sleep(1)

        # All attempts failed
        raise Exception(f"Structured generation failed after {max_retries} attempts: {last_error}")
