"""
CSV parser for PubMed data with robust column detection and batch processing.

Features:
- Automatic column name detection
- Batch iteration for memory efficiency
- Resume capability from checkpoints
- Flexible handling of various CSV formats
"""

from typing import Iterator, List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
from loguru import logger

from .models import Article
from .config import Config


class CSVParser:
    """Parser for PubMed CSV files with flexible column detection."""

    def __init__(self, config: Config):
        """
        Initialize CSV parser.

        Args:
            config: Application configuration
        """
        self.config = config
        self.csv_path = config.input_csv
        self.df: Optional[pd.DataFrame] = None
        self.total_rows: int = 0

        # Detected column mappings
        self.abstract_col: Optional[str] = None
        self.pmid_col: Optional[str] = None
        self.title_col: Optional[str] = None
        self.authors_col: Optional[str] = None
        self.journal_col: Optional[str] = None
        self.year_col: Optional[str] = None
        self.doi_col: Optional[str] = None

    def load_and_validate(self) -> None:
        """
        Load CSV file and detect column mappings.

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns cannot be detected
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        logger.info(f"Loading CSV file: {self.csv_path}")

        # Load CSV
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Try alternative encodings
            logger.warning("UTF-8 decoding failed, trying latin-1")
            self.df = pd.read_csv(self.csv_path, encoding='latin-1')

        self.total_rows = len(self.df)
        logger.info(f"Loaded {self.total_rows} rows")

        # Detect columns
        self._detect_columns()

        # Validate required columns
        if not self.abstract_col:
            raise ValueError(
                f"Could not detect abstract column. Available columns: {list(self.df.columns)}"
            )

        if not self.pmid_col:
            raise ValueError(
                f"Could not detect PMID column. Available columns: {list(self.df.columns)}"
            )

        logger.info(f"Detected columns:")
        logger.info(f"  PMID: {self.pmid_col}")
        logger.info(f"  Abstract: {self.abstract_col}")
        if self.title_col:
            logger.info(f"  Title: {self.title_col}")
        if self.authors_col:
            logger.info(f"  Authors: {self.authors_col}")
        if self.journal_col:
            logger.info(f"  Journal: {self.journal_col}")
        if self.year_col:
            logger.info(f"  Year: {self.year_col}")

    def _detect_columns(self) -> None:
        """Detect column names from the dataframe."""
        if self.df is None:
            return

        columns = list(self.df.columns)
        columns_lower = [c.lower() for c in columns]

        # Detect abstract column
        for candidate in self.config.abstract_column_names:
            candidate_lower = candidate.lower()
            if candidate_lower in columns_lower:
                idx = columns_lower.index(candidate_lower)
                self.abstract_col = columns[idx]
                break

        # Detect PMID column
        for candidate in self.config.pmid_column_names:
            candidate_lower = candidate.lower()
            if candidate_lower in columns_lower:
                idx = columns_lower.index(candidate_lower)
                self.pmid_col = columns[idx]
                break

        # Detect title column
        for candidate in self.config.title_column_names:
            candidate_lower = candidate.lower()
            if candidate_lower in columns_lower:
                idx = columns_lower.index(candidate_lower)
                self.title_col = columns[idx]
                break

        # Detect other columns (best effort)
        for col in columns:
            col_lower = col.lower()

            if not self.authors_col and any(kw in col_lower for kw in ['author', 'authors']):
                self.authors_col = col

            if not self.journal_col and any(kw in col_lower for kw in ['journal', 'source']):
                self.journal_col = col

            if not self.year_col and any(kw in col_lower for kw in ['year', 'pubdate', 'publication_year']):
                self.year_col = col

            if not self.doi_col and 'doi' in col_lower:
                self.doi_col = col

    def _row_to_article(self, row: pd.Series, index: int) -> Optional[Article]:
        """
        Convert a dataframe row to an Article object.

        Args:
            row: Pandas series representing a row
            index: Row index

        Returns:
            Article object or None if conversion fails
        """
        try:
            # Extract required fields
            pmid = str(row[self.pmid_col]) if pd.notna(row[self.pmid_col]) else f"row_{index}"
            abstract = str(row[self.abstract_col]) if pd.notna(row[self.abstract_col]) else ""

            # Skip if abstract is empty
            if not abstract.strip():
                logger.warning(f"Row {index}: Empty abstract, skipping")
                return None

            # Extract optional fields
            title = str(row[self.title_col]) if self.title_col and pd.notna(row[self.title_col]) else None
            authors = str(row[self.authors_col]) if self.authors_col and pd.notna(row[self.authors_col]) else None
            journal = str(row[self.journal_col]) if self.journal_col and pd.notna(row[self.journal_col]) else None
            doi = str(row[self.doi_col]) if self.doi_col and pd.notna(row[self.doi_col]) else None

            # Parse year
            year = None
            if self.year_col and pd.notna(row[self.year_col]):
                try:
                    year_val = row[self.year_col]
                    if isinstance(year_val, (int, float)):
                        year = int(year_val)
                    else:
                        # Try to extract year from string (e.g., "2020-01-15")
                        year_str = str(year_val)
                        year = int(year_str[:4]) if len(year_str) >= 4 else None
                except (ValueError, TypeError):
                    logger.warning(f"Row {index}: Could not parse year: {row[self.year_col]}")

            return Article(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                doi=doi
            )

        except Exception as e:
            logger.error(f"Row {index}: Error converting to Article: {e}")
            return None

    def iterate_articles(
        self,
        start_index: int = 0,
        batch_size: Optional[int] = None
    ) -> Iterator[tuple[int, Article]]:
        """
        Iterate over articles in batches.

        Args:
            start_index: Index to start from (for resume capability)
            batch_size: Number of articles per batch (uses config if not specified)

        Yields:
            Tuple of (index, Article)
        """
        if self.df is None:
            raise RuntimeError("CSV not loaded. Call load_and_validate() first.")

        batch_size = batch_size or self.config.batch_size

        for idx in range(start_index, len(self.df)):
            row = self.df.iloc[idx]
            article = self._row_to_article(row, idx)

            if article is not None:
                yield idx, article

    def get_articles_batch(
        self,
        start_index: int,
        batch_size: int
    ) -> List[tuple[int, Article]]:
        """
        Get a batch of articles.

        Args:
            start_index: Starting index
            batch_size: Number of articles to retrieve

        Returns:
            List of (index, Article) tuples
        """
        if self.df is None:
            raise RuntimeError("CSV not loaded. Call load_and_validate() first.")

        batch = []
        end_index = min(start_index + batch_size, len(self.df))

        for idx in range(start_index, end_index):
            row = self.df.iloc[idx]
            article = self._row_to_article(row, idx)

            if article is not None:
                batch.append((idx, article))

        return batch

    def get_article_by_index(self, index: int) -> Optional[Article]:
        """
        Get a single article by index.

        Args:
            index: Row index

        Returns:
            Article or None if conversion fails
        """
        if self.df is None:
            raise RuntimeError("CSV not loaded. Call load_and_validate() first.")

        if index < 0 or index >= len(self.df):
            raise IndexError(f"Index {index} out of range [0, {len(self.df)})")

        row = self.df.iloc[index]
        return self._row_to_article(row, index)

    def get_total_count(self) -> int:
        """Get total number of articles in the CSV."""
        return self.total_rows

    def get_column_info(self) -> Dict[str, Any]:
        """Get information about detected columns."""
        return {
            'total_rows': self.total_rows,
            'columns': {
                'pmid': self.pmid_col,
                'abstract': self.abstract_col,
                'title': self.title_col,
                'authors': self.authors_col,
                'journal': self.journal_col,
                'year': self.year_col,
                'doi': self.doi_col,
            },
            'all_columns': list(self.df.columns) if self.df is not None else []
        }
