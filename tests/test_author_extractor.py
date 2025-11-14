import pytest
from src.network.author_extractor import AuthorExtractor

@pytest.fixture
def extractor():
    return AuthorExtractor()

@pytest.fixture
def sample_article():
    return {
        'pmid': '12345678',
        'title': 'Novel CAR-T Therapy',
        'year': '2023',
        'authors': [
            {
                'last_name': 'Smith',
                'first_name': 'John',
                'initials': 'J',
                'affiliations': ["Boston Children's Hospital, Boston, MA, USA"]
            },
            {
                'last_name': 'Doe',
                'first_name': 'Jane',
                'initials': 'J',
                'affiliations': ["Harvard Medical School, Boston, MA"]
            },
            {
                'last_name': 'Brown',
                'first_name': 'Alice',
                'initials': 'A',
                'affiliations': [
                    "Boston Children's Hospital, Boston, MA",
                    "Harvard Medical School, Boston, MA"
                ]
            }
        ]
    }

def test_extract_authors_from_article(extractor, sample_article):
    """Should extract all authors with metadata."""
    authors = extractor.extract_authors(sample_article)

    assert len(authors) == 3
    assert authors[0]['full_name'] == 'John Smith'
    assert authors[0]['position'] == 0  # First author
    assert authors[2]['position'] == 2  # Third author

def test_extract_affiliations(extractor, sample_article):
    """Should extract affiliations for each author."""
    authors = extractor.extract_authors(sample_article)

    # First author
    assert len(authors[0]['affiliations']) == 1
    assert "Boston Children's Hospital" in authors[0]['affiliations'][0]['institution']

    # Third author (multiple affiliations)
    assert len(authors[2]['affiliations']) == 2

def test_extract_co_authors(extractor, sample_article):
    """Should identify co-author relationships."""
    co_authors = extractor.extract_co_authorship(sample_article)

    # Should have 3 pairs: (Smith,Doe), (Smith,Brown), (Doe,Brown)
    assert len(co_authors) == 3

    # Check structure
    assert co_authors[0]['author1'] == 'John Smith'
    assert co_authors[0]['author2'] == 'Jane Doe'
    assert co_authors[0]['pmid'] == '12345678'
    assert co_authors[0]['year'] == '2023'

def test_identify_first_and_last_authors(extractor, sample_article):
    """Should mark first and last author positions."""
    authors = extractor.extract_authors(sample_article)

    assert authors[0]['is_first_author'] is True
    assert authors[1]['is_first_author'] is False
    assert authors[2]['is_last_author'] is True

def test_handle_missing_author_info(extractor):
    """Should handle articles with incomplete author data."""
    article = {
        'pmid': '99999',
        'authors': [
            {'last_name': 'Unknown', 'first_name': ''},
            {'last_name': '', 'first_name': 'Mystery'}
        ]
    }

    authors = extractor.extract_authors(article)

    # Should still create author entries
    assert len(authors) == 2

def test_create_author_id(extractor):
    """Should create consistent author identifiers."""
    author1 = {
        'last_name': 'Smith',
        'first_name': 'John',
        'affiliations': ["Boston Children's Hospital"]
    }

    author2 = {
        'last_name': 'Smith',
        'first_name': 'John',
        'affiliations': ["Boston Children's Hospital"]
    }

    id1 = extractor.create_author_id(author1)
    id2 = extractor.create_author_id(author2)

    # Same author should get same ID
    assert id1 == id2
