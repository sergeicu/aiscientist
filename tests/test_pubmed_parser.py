"""
Tests for PubMedParser class.

Tests XML parsing of PubMed article metadata.
"""

import pytest
from pathlib import Path
from src.scrapers.pubmed_parser import PubMedParser


@pytest.fixture
def sample_article_xml():
    """Load sample PubMed XML."""
    return '''
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">12345678</PMID>
            <Article PubModel="Print">
                <ArticleTitle>A Novel CAR-T Therapy for Pediatric Leukemia</ArticleTitle>
                <Abstract>
                    <AbstractText>This study demonstrates...</AbstractText>
                </Abstract>
                <AuthorList CompleteYN="Y">
                    <Author ValidYN="Y">
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                        <Initials>J</Initials>
                        <AffiliationInfo>
                            <Affiliation>Boston Children's Hospital, Boston, MA, USA</Affiliation>
                        </AffiliationInfo>
                    </Author>
                    <Author ValidYN="Y">
                        <LastName>Doe</LastName>
                        <ForeName>Jane</ForeName>
                        <Initials>J</Initials>
                        <AffiliationInfo>
                            <Affiliation>Harvard Medical School, Boston, MA, USA</Affiliation>
                        </AffiliationInfo>
                    </Author>
                </AuthorList>
                <Journal>
                    <Title>Nature Medicine</Title>
                    <JournalIssue CitedMedium="Internet">
                        <Volume>29</Volume>
                        <Issue>3</Issue>
                        <PubDate>
                            <Year>2023</Year>
                            <Month>Mar</Month>
                        </PubDate>
                    </JournalIssue>
                </Journal>
                <ELocationID EIdType="doi" ValidYN="Y">10.1038/s41591-023-12345-6</ELocationID>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    '''


def test_parse_basic_metadata(sample_article_xml):
    """Should extract PMID, title, abstract."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert article['pmid'] == '12345678'
    assert 'CAR-T Therapy' in article['title']
    assert 'demonstrates' in article['abstract']


def test_parse_authors(sample_article_xml):
    """Should extract all authors with affiliations."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert len(article['authors']) == 2
    assert article['authors'][0]['last_name'] == 'Smith'
    assert article['authors'][0]['first_name'] == 'John'
    assert "Boston Children's Hospital" in article['authors'][0]['affiliations'][0]


def test_parse_journal_info(sample_article_xml):
    """Should extract journal, volume, issue, date."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert article['journal'] == 'Nature Medicine'
    assert article['year'] == '2023'
    assert article['volume'] == '29'


def test_parse_doi(sample_article_xml):
    """Should extract DOI."""
    parser = PubMedParser()
    article = parser.parse_article(sample_article_xml)

    assert article['doi'] == '10.1038/s41591-023-12345-6'


def test_handle_missing_fields():
    """Should handle articles with missing fields gracefully."""
    minimal_xml = '''
    <PubmedArticle>
        <MedlineCitation>
            <PMID>99999</PMID>
            <Article>
                <ArticleTitle>Minimal Article</ArticleTitle>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    '''

    parser = PubMedParser()
    article = parser.parse_article(minimal_xml)

    assert article['pmid'] == '99999'
    assert article['abstract'] == ''
    assert article['authors'] == []
    assert article['doi'] is None


def test_parse_invalid_xml():
    """Should raise ValueError for invalid XML."""
    parser = PubMedParser()

    with pytest.raises(ValueError, match="Invalid XML"):
        parser.parse_article("<invalid>xml")


def test_parse_missing_structure():
    """Should raise ValueError for missing required structure."""
    parser = PubMedParser()
    invalid_structure = '<root><something>else</something></root>'

    with pytest.raises(ValueError, match="Invalid PubMed XML structure"):
        parser.parse_article(invalid_structure)


def test_parse_article_from_record():
    """Should parse article from Entrez record format."""
    parser = PubMedParser()

    record = {
        'MedlineCitation': {
            'PMID': '12345',
            'Article': {
                'ArticleTitle': 'Test Article',
                'Abstract': {
                    'AbstractText': ['First part', 'Second part']
                },
                'Journal': {
                    'Title': 'Test Journal',
                    'JournalIssue': {
                        'Volume': '10',
                        'PubDate': {
                            'Year': '2023'
                        }
                    }
                },
                'AuthorList': [
                    {
                        'LastName': 'Doe',
                        'ForeName': 'John',
                        'Initials': 'J',
                        'AffiliationInfo': [
                            {'Affiliation': 'Test University'}
                        ]
                    }
                ],
                'ELocationID': []
            },
            'MeshHeadingList': [
                {'DescriptorName': 'Cancer'}
            ]
        }
    }

    article = parser.parse_article_from_record(record)

    assert article['pmid'] == '12345'
    assert article['title'] == 'Test Article'
    assert 'First part' in article['abstract']
    assert article['journal'] == 'Test Journal'
    assert article['year'] == '2023'
    assert len(article['authors']) == 1
    assert article['authors'][0]['last_name'] == 'Doe'
    assert len(article['mesh_terms']) == 1
