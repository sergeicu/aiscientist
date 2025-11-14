import pytest
from src.network.affiliation_parser import AffiliationParser

@pytest.fixture
def parser():
    return AffiliationParser()

def test_parse_simple_affiliation(parser):
    """Should parse simple affiliation string."""
    affil = "Boston Children's Hospital, Boston, MA, USA"

    result = parser.parse(affil)

    assert result['institution'] == "Boston Children's Hospital"
    assert result['city'] == 'Boston'
    assert result['state'] == 'MA'
    assert result['country'] == 'USA'

def test_parse_with_department(parser):
    """Should extract department information."""
    affil = "Department of Pediatrics, Harvard Medical School, Boston, MA"

    result = parser.parse(affil)

    assert result['department'] == 'Department of Pediatrics'
    assert result['institution'] == 'Harvard Medical School'

def test_parse_multiple_institutions(parser):
    """Should handle multiple institutions in one string."""
    affil = "Dept of Medicine, Hospital A; Dept of Surgery, Hospital B"

    results = parser.parse_multiple(affil)

    assert len(results) == 2
    assert 'Hospital A' in results[0]['institution']
    assert 'Hospital B' in results[1]['institution']

def test_normalize_institution_name(parser):
    """Should normalize institution names."""
    assert parser.normalize("BCH") == "Boston Children's Hospital"
    assert parser.normalize("MGH") == "Massachusetts General Hospital"
    assert parser.normalize("Harvard Med School") == "Harvard Medical School"

def test_extract_country(parser):
    """Should extract country from various formats."""
    assert parser.extract_country("Boston, MA, USA") == "USA"
    assert parser.extract_country("London, UK") == "UK"
    assert parser.extract_country("Beijing, China") == "China"

def test_handle_malformed_affiliation(parser):
    """Should handle poorly formatted strings gracefully."""
    affil = "Some random text without clear structure"

    result = parser.parse(affil)

    # Should still extract what it can
    assert result['institution'] is not None or result['raw'] == affil
