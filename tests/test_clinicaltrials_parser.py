"""Tests for ClinicalTrials.gov parser."""

import pytest
from pathlib import Path
import json
from src.scrapers.clinicaltrials_parser import ClinicalTrialsParser


@pytest.fixture
def sample_study():
    """Sample study from ClinicalTrials.gov API."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "A Phase 2 Trial of Novel CAR-T Therapy"
            },
            "statusModule": {
                "statusVerifiedDate": "2024-01",
                "overallStatus": "RECRUITING",
                "startDateStruct": {
                    "date": "2024-01-15"
                }
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {
                    "name": "Boston Children's Hospital"
                },
                "collaborators": [
                    {"name": "Harvard Medical School"}
                ]
            },
            "designModule": {
                "phases": ["PHASE2"],
                "studyType": "INTERVENTIONAL",
                "enrollmentInfo": {
                    "count": 50
                }
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "type": "BIOLOGICAL",
                        "name": "CAR-T Cell Therapy"
                    }
                ]
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Overall Response Rate",
                        "timeFrame": "6 months"
                    }
                ],
                "secondaryOutcomes": [
                    {
                        "measure": "Progression-Free Survival",
                        "timeFrame": "12 months"
                    }
                ]
            },
            "conditionsModule": {
                "conditions": ["Pediatric Leukemia", "ALL"]
            },
            "descriptionModule": {
                "briefSummary": "This study evaluates CAR-T therapy..."
            }
        }
    }


def test_parse_basic_metadata(sample_study):
    """Should extract NCT ID, title, status."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert trial['nct_id'] == 'NCT12345678'
    assert 'CAR-T Therapy' in trial['title']
    assert trial['status'] == 'RECRUITING'


def test_parse_sponsor_info(sample_study):
    """Should extract lead sponsor and collaborators."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert trial['lead_sponsor'] == "Boston Children's Hospital"
    assert len(trial['collaborators']) == 1
    assert trial['collaborators'][0] == "Harvard Medical School"


def test_parse_phase_and_type(sample_study):
    """Should extract trial phase and study type."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert trial['phase'] == 'Phase 2'
    assert trial['study_type'] == 'INTERVENTIONAL'
    assert trial['enrollment'] == 50


def test_parse_interventions(sample_study):
    """Should extract intervention details."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert len(trial['interventions']) == 1
    assert trial['interventions'][0]['type'] == 'BIOLOGICAL'
    assert trial['interventions'][0]['name'] == 'CAR-T Cell Therapy'


def test_parse_outcomes(sample_study):
    """Should extract primary and secondary outcomes."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert len(trial['primary_outcomes']) == 1
    assert 'Response Rate' in trial['primary_outcomes'][0]['measure']

    assert len(trial['secondary_outcomes']) == 1
    assert 'Survival' in trial['secondary_outcomes'][0]['measure']


def test_parse_conditions(sample_study):
    """Should extract conditions/diseases."""
    parser = ClinicalTrialsParser()
    trial = parser.parse_study(sample_study)

    assert 'Pediatric Leukemia' in trial['conditions']
    assert 'ALL' in trial['conditions']


def test_handle_missing_fields():
    """Should handle studies with minimal data."""
    minimal_study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT99999999",
                "briefTitle": "Minimal Study"
            },
            "statusModule": {
                "overallStatus": "COMPLETED"
            }
        }
    }

    parser = ClinicalTrialsParser()
    trial = parser.parse_study(minimal_study)

    assert trial['nct_id'] == 'NCT99999999'
    assert trial['lead_sponsor'] is None
    assert trial['interventions'] == []
    assert trial['enrollment'] is None


def test_normalize_phase():
    """Should normalize phase names."""
    parser = ClinicalTrialsParser()

    assert parser.normalize_phase(['PHASE1']) == 'Phase 1'
    assert parser.normalize_phase(['PHASE2', 'PHASE3']) == 'Phase 2/Phase 3'
    assert parser.normalize_phase(['NA']) == 'Not Applicable'
    assert parser.normalize_phase([]) is None
