import re
from typing import Dict, List, Optional
from thefuzz import fuzz
from loguru import logger

class AffiliationParser:
    """
    Parse affiliation strings to extract structured information.

    Handles various affiliation formats from PubMed.
    Extracts: institution, department, city, state, country.

    Example:
        >>> parser = AffiliationParser()
        >>> result = parser.parse(
        ...     "Dept of Pediatrics, Boston Children's Hospital, Boston, MA, USA"
        ... )
        >>> result['institution']
        "Boston Children's Hospital"
    """

    # Common institution abbreviations
    INSTITUTION_MAP = {
        'BCH': "Boston Children's Hospital",
        'MGH': 'Massachusetts General Hospital',
        'BWH': 'Brigham and Women\'s Hospital',
        'HMS': 'Harvard Medical School',
        'MIT': 'Massachusetts Institute of Technology',
        'NIH': 'National Institutes of Health',
        'CDC': 'Centers for Disease Control and Prevention'
    }

    # Country patterns
    COUNTRY_PATTERNS = [
        r',\s*(USA|United States)$',
        r',\s*(UK|United Kingdom)$',
        r',\s*(China)$',
        r',\s*(Japan)$',
        r',\s*(Germany)$',
        r',\s*(France)$',
        r',\s*(Canada)$'
    ]

    # Department keywords
    DEPT_KEYWORDS = [
        'Department', 'Dept', 'Division', 'Div',
        'Center', 'Centre', 'Institute', 'School'
    ]

    def parse(self, affiliation: str) -> Dict:
        """
        Parse affiliation string to structured dict.

        Args:
            affiliation: Raw affiliation string

        Returns:
            Dictionary with keys: institution, department, city, state, country, raw
        """
        if not affiliation:
            return self._empty_result(affiliation)

        # Split by common separators
        parts = re.split(r'[;,]\s*', affiliation)

        # Extract components
        institution = None
        department = None
        city = None
        state = None
        country = self.extract_country(affiliation)

        # Identify department (usually first part)
        if parts and any(kw in parts[0] for kw in self.DEPT_KEYWORDS):
            department = parts[0].strip()
            remaining_parts = parts[1:]
        else:
            remaining_parts = parts

        # Identify institution (usually has "Hospital", "University", "Institute")
        institution_keywords = ['Hospital', 'University', 'Institute', 'College', 'School']

        for part in remaining_parts:
            if any(kw in part for kw in institution_keywords):
                institution = part.strip()
                break

        # Extract city and state (usually last 2-3 parts before country)
        if len(remaining_parts) >= 2:
            # Common pattern: "City, STATE" or "City, STATE, Country"
            state_pattern = r'\b[A-Z]{2}\b'  # Two capital letters (US states)

            for i, part in enumerate(remaining_parts[-3:]):
                if re.search(state_pattern, part):
                    state = re.search(state_pattern, part).group()
                    # Calculate the actual index in remaining_parts
                    actual_index = len(remaining_parts) - 3 + i
                    if actual_index > 0:
                        city = remaining_parts[actual_index - 1].strip()

        return {
            'institution': institution,
            'department': department,
            'city': city,
            'state': state,
            'country': country,
            'raw': affiliation
        }

    def parse_multiple(self, affiliation: str) -> List[Dict]:
        """
        Parse affiliation string with multiple institutions.

        Some authors list multiple affiliations separated by semicolons.

        Args:
            affiliation: Affiliation string (may contain multiple)

        Returns:
            List of parsed affiliation dicts
        """
        # Split by semicolon (common separator for multiple affiliations)
        affiliations = affiliation.split(';')

        return [self.parse(a.strip()) for a in affiliations if a.strip()]

    def normalize(self, name: str) -> str:
        """
        Normalize institution name.

        Expands abbreviations and standardizes names.

        Args:
            name: Institution name or abbreviation

        Returns:
            Normalized name
        """
        # Check if it's a known abbreviation
        if name in self.INSTITUTION_MAP:
            return self.INSTITUTION_MAP[name]

        # Remove common suffixes for matching
        normalized = name.strip()

        # Fuzzy match against known institutions
        best_match = None
        best_score = 0

        for abbr, full_name in self.INSTITUTION_MAP.items():
            score = fuzz.ratio(normalized.lower(), full_name.lower())
            if score > best_score:
                best_score = score
                best_match = full_name

        # Use fuzzy match if score > 85
        if best_score > 85:
            return best_match

        return normalized

    def extract_country(self, affiliation: str) -> Optional[str]:
        """
        Extract country from affiliation string.

        Args:
            affiliation: Affiliation string

        Returns:
            Country name or None
        """
        for pattern in self.COUNTRY_PATTERNS:
            match = re.search(pattern, affiliation)
            if match:
                return match.group(1)

        # Check last part
        parts = affiliation.split(',')
        if parts:
            last_part = parts[-1].strip()
            # If last part is short and capitalized, likely country
            if len(last_part) < 30 and last_part[0].isupper():
                return last_part

        return None

    def _empty_result(self, raw: str) -> Dict:
        """Return empty result dict."""
        return {
            'institution': None,
            'department': None,
            'city': None,
            'state': None,
            'country': None,
            'raw': raw
        }
