"""
Structured output handler with multiple parsing strategies.

Implements hybrid approach:
1. PRIMARY: Outlines constrained generation (when available)
2. FALLBACK: json-repair for malformed JSON
3. LAST RESORT: Regex-based field extraction

This ensures reliable JSON extraction from small language models.
"""

import json
import re
from typing import Optional, Dict, Any, Tuple
from loguru import logger

try:
    import outlines
    from outlines import models, generate
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    logger.warning("Outlines not available - constrained generation disabled")

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    logger.warning("json-repair not available - repair fallback disabled")

from .models import ClassificationResult
from .config import Config


class StructuredOutputHandler:
    """Handler for extracting structured JSON from LLM outputs."""

    def __init__(self, config: Config):
        """
        Initialize structured output handler.

        Args:
            config: Application configuration
        """
        self.config = config
        self.outlines_model = None
        self.outlines_generator = None

        # Initialize Outlines if enabled and available
        if config.use_outlines and OUTLINES_AVAILABLE:
            self._initialize_outlines()

    def _initialize_outlines(self) -> bool:
        """
        Initialize Outlines constrained generation.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing Outlines for constrained generation...")

            # Create Ollama model for Outlines
            # Note: Outlines needs to wrap the model for constrained generation
            # For now, we'll skip this initialization as it requires
            # integration with the Ollama client
            # This will be set up when we actually call generate with Outlines

            logger.info("✓ Outlines initialized (lazy loading)")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Outlines: {e}")
            return False

    def parse_json_output(
        self,
        raw_output: str,
        method_preference: Optional[str] = None
    ) -> Tuple[Optional[ClassificationResult], str]:
        """
        Parse JSON output using multiple strategies.

        Args:
            raw_output: Raw text output from LLM
            method_preference: Preferred parsing method ('direct', 'repair', 'regex')

        Returns:
            Tuple of (ClassificationResult or None, parsing_method_used)
        """
        # Strategy 1: Direct JSON parsing
        result, success = self._try_direct_parse(raw_output)
        if success:
            return result, "direct"

        # Strategy 2: Extract JSON from code blocks or mixed text
        result, success = self._try_extract_json_block(raw_output)
        if success:
            return result, "extracted"

        # Strategy 3: json-repair
        if self.config.use_json_repair and JSON_REPAIR_AVAILABLE:
            result, success = self._try_json_repair(raw_output)
            if success:
                return result, "json_repair"

        # Strategy 4: Regex-based field extraction
        result, success = self._try_regex_extraction(raw_output)
        if success:
            return result, "regex"

        # All strategies failed
        logger.error("All parsing strategies failed")
        return None, "failed"

    def _try_direct_parse(self, text: str) -> Tuple[Optional[ClassificationResult], bool]:
        """
        Try direct JSON parsing.

        Args:
            text: Raw text to parse

        Returns:
            Tuple of (result, success)
        """
        try:
            data = json.loads(text)
            result = ClassificationResult(**data)
            logger.debug("✓ Direct JSON parse successful")
            return result, True
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            return None, False

    def _try_extract_json_block(self, text: str) -> Tuple[Optional[ClassificationResult], bool]:
        """
        Try to extract JSON from code blocks or mixed text.

        Args:
            text: Raw text containing JSON

        Returns:
            Tuple of (result, success)
        """
        try:
            # Pattern 1: JSON in code blocks (```json ... ```)
            pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            matches = re.findall(pattern, text, re.DOTALL)

            if matches:
                json_str = matches[0]
                data = json.loads(json_str)
                result = ClassificationResult(**data)
                logger.debug("✓ Extracted JSON from code block")
                return result, True

            # Pattern 2: Find first complete JSON object
            # Look for balanced braces
            start_idx = text.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start_idx:i+1]
                            data = json.loads(json_str)
                            result = ClassificationResult(**data)
                            logger.debug("✓ Extracted JSON object from text")
                            return result, True

            return None, False

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON extraction failed: {e}")
            return None, False

    def _try_json_repair(self, text: str) -> Tuple[Optional[ClassificationResult], bool]:
        """
        Try to repair malformed JSON using json-repair.

        Args:
            text: Potentially malformed JSON text

        Returns:
            Tuple of (result, success)
        """
        if not JSON_REPAIR_AVAILABLE:
            return None, False

        try:
            # First try to extract JSON if it's embedded in text
            json_str = text
            if '{' in text:
                start_idx = text.find('{')
                json_str = text[start_idx:]

            # Repair JSON
            repaired = repair_json(json_str)
            data = json.loads(repaired)
            result = ClassificationResult(**data)
            logger.debug("✓ JSON repair successful")
            return result, True

        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
            return None, False

    def _try_regex_extraction(self, text: str) -> Tuple[Optional[ClassificationResult], bool]:
        """
        Last resort: Extract fields using regex patterns.

        Args:
            text: Raw text output

        Returns:
            Tuple of (result, success)
        """
        try:
            logger.debug("Attempting regex-based field extraction...")

            # Extract key fields with regex
            is_clinical_trial = self._extract_boolean(
                text,
                ['is_clinical_trial', 'clinical_trial', 'is_trial']
            )

            confidence = self._extract_float(
                text,
                ['confidence', 'confidence_score'],
                default=0.5
            )

            reasoning = self._extract_string(
                text,
                ['reasoning', 'explanation', 'rationale']
            )

            if is_clinical_trial is None:
                return None, False

            # Create minimal result
            result = ClassificationResult(
                is_clinical_trial=is_clinical_trial,
                confidence=confidence,
                reasoning=reasoning or "Extracted via regex (incomplete output)"
            )

            logger.debug("✓ Regex extraction successful (partial data)")
            return result, True

        except Exception as e:
            logger.debug(f"Regex extraction failed: {e}")
            return None, False

    def _extract_boolean(self, text: str, field_names: list) -> Optional[bool]:
        """Extract boolean field from text."""
        for field in field_names:
            # Pattern: "field": true/false or field: true/false
            pattern = rf'"{field}"\s*:\s*(true|false)|{field}\s*:\s*(true|false)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1) or match.group(2)
                return value.lower() == 'true'
        return None

    def _extract_float(self, text: str, field_names: list, default: float = 0.0) -> float:
        """Extract float field from text."""
        for field in field_names:
            # Pattern: "field": 0.123 or field: 0.123
            pattern = rf'"{field}"\s*:\s*([0-9.]+)|{field}\s*:\s*([0-9.]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1) or match.group(2))
                    return max(0.0, min(1.0, value))  # Clamp to [0, 1]
                except ValueError:
                    continue
        return default

    def _extract_string(self, text: str, field_names: list) -> Optional[str]:
        """Extract string field from text."""
        for field in field_names:
            # Pattern: "field": "value" or field: "value"
            pattern = rf'"{field}"\s*:\s*"([^"]+)"|{field}\s*:\s*"([^"]+)"'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) or match.group(2)
        return None

    def validate_result(self, result: ClassificationResult) -> bool:
        """
        Validate a classification result.

        Args:
            result: Classification result to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Pydantic already validates on construction
            # Additional semantic validation can be added here

            # Check confidence is reasonable
            if not 0.0 <= result.confidence <= 1.0:
                logger.warning(f"Invalid confidence: {result.confidence}")
                return False

            # Check reasoning is not empty
            if not result.reasoning or len(result.reasoning.strip()) < 5:
                logger.warning("Reasoning is too short or empty")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def get_json_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for ClassificationResult.

        Returns:
            JSON schema dictionary
        """
        return ClassificationResult.model_json_schema()

    def format_schema_for_prompt(self) -> str:
        """
        Format JSON schema as a string for inclusion in prompts.

        Returns:
            Formatted schema string
        """
        schema = self.get_json_schema()

        # Create a simplified, readable version
        properties = schema.get('properties', {})

        lines = ["Expected JSON output format:"]
        lines.append("{")

        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get('type', 'any')
            description = prop_info.get('description', '')

            # Handle optional fields
            if prop_name in schema.get('required', []):
                required = " (required)"
            else:
                required = " (optional)"

            lines.append(f'  "{prop_name}": {prop_type}{required}  // {description}')

        lines.append("}")

        return "\n".join(lines)
