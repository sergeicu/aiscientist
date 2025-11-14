from typing import List, Dict
import hashlib
from loguru import logger
from .affiliation_parser import AffiliationParser

class AuthorExtractor:
    """
    Extract author information and co-authorship relationships from articles.

    Creates structured author records and identifies collaboration patterns.

    Example:
        >>> extractor = AuthorExtractor()
        >>> authors = extractor.extract_authors(article)
        >>> co_authors = extractor.extract_co_authorship(article)
    """

    def __init__(self):
        self.affiliation_parser = AffiliationParser()

    def extract_authors(self, article: Dict) -> List[Dict]:
        """
        Extract author information from article.

        Args:
            article: Article dictionary with 'authors' field

        Returns:
            List of author dictionaries with enriched metadata
        """
        authors = []
        author_list = article.get('authors', [])
        total_authors = len(author_list)

        for i, author in enumerate(author_list):
            # Basic info
            last_name = author.get('last_name', '')
            first_name = author.get('first_name', '')
            initials = author.get('initials', '')

            full_name = f"{first_name} {last_name}".strip()
            if not full_name:
                full_name = f"{initials} {last_name}".strip()

            # Parse affiliations
            affiliations = []
            for affil_str in author.get('affiliations', []):
                parsed = self.affiliation_parser.parse(affil_str)
                affiliations.append(parsed)

            # Position metadata
            is_first = (i == 0)
            is_last = (i == total_authors - 1)

            # Create author record
            author_record = {
                'full_name': full_name,
                'last_name': last_name,
                'first_name': first_name,
                'initials': initials,
                'position': i,
                'is_first_author': is_first,
                'is_last_author': is_last,
                'affiliations': affiliations,
                'pmid': article.get('pmid'),
                'year': article.get('year'),
                'author_id': self.create_author_id({
                    'last_name': last_name,
                    'first_name': first_name,
                    'affiliations': author.get('affiliations', [])
                })
            }

            authors.append(author_record)

        return authors

    def extract_co_authorship(self, article: Dict) -> List[Dict]:
        """
        Extract co-authorship relationships from article.

        Creates pairwise collaborations between all authors.

        Args:
            article: Article dictionary

        Returns:
            List of co-author relationship dicts
        """
        authors = self.extract_authors(article)
        co_authorships = []

        # Create pairwise combinations
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                author1 = authors[i]
                author2 = authors[j]

                co_authorship = {
                    'author1': author1['full_name'],
                    'author1_id': author1['author_id'],
                    'author2': author2['full_name'],
                    'author2_id': author2['author_id'],
                    'pmid': article.get('pmid'),
                    'year': article.get('year'),
                    'title': article.get('title', '')
                }

                co_authorships.append(co_authorship)

        return co_authorships

    def create_author_id(self, author: Dict) -> str:
        """
        Create unique author identifier.

        Uses last name + first initial + primary affiliation.
        This is a simple disambiguation strategy.

        Args:
            author: Author dictionary

        Returns:
            Author ID string
        """
        last_name = author.get('last_name', '').lower()
        first_name = author.get('first_name', '')
        first_initial = first_name[0].lower() if first_name else ''

        # Use primary affiliation if available
        affiliations = author.get('affiliations', [])
        affiliation = ''
        if affiliations:
            # Parse first affiliation
            if isinstance(affiliations[0], str):
                parsed = self.affiliation_parser.parse(affiliations[0])
                affiliation = parsed.get('institution', '') or ''

        # Create ID: lastname_fi_institution
        id_string = f"{last_name}_{first_initial}_{affiliation}"

        # Hash for consistent length
        return hashlib.md5(id_string.encode()).hexdigest()[:16]

    def extract_institutional_collaborations(
        self,
        articles: List[Dict]
    ) -> List[Dict]:
        """
        Extract institution-level collaborations across articles.

        Identifies which institutions collaborate most frequently.

        Args:
            articles: List of article dictionaries

        Returns:
            List of institution collaboration records
        """
        collaborations = {}

        for article in articles:
            authors = self.extract_authors(article)

            # Get unique institutions from article
            institutions = set()
            for author in authors:
                for affil in author['affiliations']:
                    inst = affil.get('institution')
                    if inst:
                        institutions.add(inst)

            # Create pairwise collaborations
            inst_list = list(institutions)
            for i in range(len(inst_list)):
                for j in range(i + 1, len(inst_list)):
                    inst1, inst2 = sorted([inst_list[i], inst_list[j]])
                    key = (inst1, inst2)

                    if key not in collaborations:
                        collaborations[key] = {
                            'institution1': inst1,
                            'institution2': inst2,
                            'count': 0,
                            'pmids': []
                        }

                    collaborations[key]['count'] += 1
                    collaborations[key]['pmids'].append(article['pmid'])

        return list(collaborations.values())
