"""
PubMed XML parser.

Parse PubMed XML responses to structured dictionaries.
Handles both ESearch and EFetch XML formats.
"""

from typing import Dict, List, Optional
from xml.etree import ElementTree as ET
from loguru import logger


class PubMedParser:
    """
    Parse PubMed XML responses to structured dictionaries.

    Handles both ESearch and EFetch XML formats.
    Robust to missing fields and malformed data.
    """

    @staticmethod
    def parse_article(xml_string: str) -> Dict:
        """
        Parse single PubMed article XML.

        Args:
            xml_string: XML string from efetch

        Returns:
            Dictionary with article metadata
        """
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            raise ValueError(f"Invalid XML: {e}")

        # Navigate XML structure
        article_elem = root.find('.//Article')
        medline_elem = root.find('.//MedlineCitation')

        if article_elem is None or medline_elem is None:
            raise ValueError("Invalid PubMed XML structure")

        # Extract basic metadata
        pmid = PubMedParser._get_text(medline_elem, 'PMID', '')
        title = PubMedParser._get_text(article_elem, 'ArticleTitle', '')

        # Abstract (can be multiple AbstractText elements)
        abstract_parts = article_elem.findall('.//AbstractText')
        abstract = ' '.join(
            elem.text for elem in abstract_parts if elem.text
        )

        # Journal info
        journal_elem = article_elem.find('.//Journal')
        journal = PubMedParser._get_text(journal_elem, 'Title', '') if journal_elem else ''

        pub_date = journal_elem.find('.//PubDate') if journal_elem else None
        year = PubMedParser._get_text(pub_date, 'Year', '') if pub_date else ''

        # Volume/Issue
        journal_issue = journal_elem.find('.//JournalIssue') if journal_elem else None
        volume = PubMedParser._get_text(journal_issue, 'Volume', '') if journal_issue else ''

        # Authors
        authors = PubMedParser._parse_authors(article_elem)

        # DOI
        doi = PubMedParser._extract_doi(article_elem)

        # MeSH terms
        mesh_terms = PubMedParser._extract_mesh_terms(medline_elem)

        return {
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'journal': journal,
            'year': year,
            'volume': volume,
            'doi': doi,
            'authors': authors,
            'mesh_terms': mesh_terms
        }

    @staticmethod
    def _parse_authors(article_elem: ET.Element) -> List[Dict]:
        """Extract author list with affiliations."""
        authors = []
        author_list = article_elem.find('.//AuthorList')

        if author_list is None:
            return []

        for author_elem in author_list.findall('Author'):
            author = {
                'last_name': PubMedParser._get_text(author_elem, 'LastName', ''),
                'first_name': PubMedParser._get_text(author_elem, 'ForeName', ''),
                'initials': PubMedParser._get_text(author_elem, 'Initials', ''),
                'affiliations': []
            }

            # Extract affiliations
            affil_list = author_elem.findall('.//AffiliationInfo')
            for affil_elem in affil_list:
                affil_text = PubMedParser._get_text(affil_elem, 'Affiliation', '')
                if affil_text:
                    author['affiliations'].append(affil_text)

            authors.append(author)

        return authors

    @staticmethod
    def _extract_doi(article_elem: ET.Element) -> Optional[str]:
        """Extract DOI from ELocationID elements."""
        for eloc in article_elem.findall('.//ELocationID'):
            if eloc.get('EIdType') == 'doi':
                return eloc.text
        return None

    @staticmethod
    def _extract_mesh_terms(medline_elem: ET.Element) -> List[str]:
        """Extract MeSH (Medical Subject Headings) terms."""
        mesh_list = medline_elem.find('.//MeshHeadingList')
        if mesh_list is None:
            return []

        terms = []
        for mesh_heading in mesh_list.findall('MeshHeading'):
            descriptor = mesh_heading.find('DescriptorName')
            if descriptor is not None and descriptor.text:
                terms.append(descriptor.text)

        return terms

    @staticmethod
    def _get_text(element: Optional[ET.Element], tag: str, default: str = '') -> str:
        """Safely get text from XML element."""
        if element is None:
            return default

        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()

        return default

    @staticmethod
    def parse_article_from_record(record: Dict) -> Dict:
        """Parse article from Entrez.read() record object."""
        medline = record.get("MedlineCitation", {})
        article = medline.get("Article", {})

        # PMID
        pmid = str(medline.get("PMID", ""))

        # Title
        title = article.get("ArticleTitle", "")

        # Abstract
        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(part) for part in abstract_parts)
        else:
            abstract = str(abstract_parts) if abstract_parts else ""

        # Journal
        journal = article.get("Journal", {}).get("Title", "")

        # Date
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")

        # Volume
        volume = article.get("Journal", {}).get("JournalIssue", {}).get("Volume", "")

        # Authors
        authors = []
        for author in article.get("AuthorList", []):
            author_dict = {
                "last_name": author.get("LastName", ""),
                "first_name": author.get("ForeName", ""),
                "initials": author.get("Initials", ""),
                "affiliations": []
            }

            # Affiliations
            for affil in author.get("AffiliationInfo", []):
                affil_text = affil.get("Affiliation", "")
                if affil_text:
                    author_dict["affiliations"].append(affil_text)

            authors.append(author_dict)

        # DOI
        doi = None
        for eloc_id in article.get("ELocationID", []):
            if hasattr(eloc_id, 'attributes'):
                if eloc_id.attributes.get("EIdType") == "doi":
                    doi = str(eloc_id)

        # MeSH terms
        mesh_terms = []
        for mesh_heading in medline.get("MeshHeadingList", []):
            descriptor = mesh_heading.get("DescriptorName", "")
            if descriptor:
                mesh_terms.append(str(descriptor))

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "volume": volume,
            "doi": doi,
            "authors": authors,
            "mesh_terms": mesh_terms
        }
