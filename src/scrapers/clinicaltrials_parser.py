"""Parse ClinicalTrials.gov API v2 responses."""

from typing import Dict, List, Optional
from loguru import logger


class ClinicalTrialsParser:
    """
    Parse ClinicalTrials.gov API v2 responses.

    Extracts structured metadata from study JSON.
    Handles missing fields gracefully.
    """

    # Phase normalization mapping
    PHASE_MAP = {
        'EARLY_PHASE1': 'Early Phase 1',
        'PHASE1': 'Phase 1',
        'PHASE2': 'Phase 2',
        'PHASE3': 'Phase 3',
        'PHASE4': 'Phase 4',
        'NA': 'Not Applicable'
    }

    @staticmethod
    def parse_study(study: Dict) -> Dict:
        """
        Parse study JSON to structured dictionary.

        Args:
            study: Study object from API response

        Returns:
            Normalized trial metadata
        """
        protocol = study.get('protocolSection', {})

        # Identification
        ident_module = protocol.get('identificationModule', {})
        nct_id = ident_module.get('nctId', '')
        title = ident_module.get('briefTitle', '')
        official_title = ident_module.get('officialTitle', '')

        # Status
        status_module = protocol.get('statusModule', {})
        status = status_module.get('overallStatus', '')
        start_date = ClinicalTrialsParser._extract_date(
            status_module.get('startDateStruct', {})
        )
        completion_date = ClinicalTrialsParser._extract_date(
            status_module.get('completionDateStruct', {})
        )

        # Sponsor
        sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
        lead_sponsor = None
        if 'leadSponsor' in sponsor_module:
            lead_sponsor = sponsor_module['leadSponsor'].get('name')

        collaborators = [
            c.get('name', '') for c in sponsor_module.get('collaborators', [])
        ]

        # Design
        design_module = protocol.get('designModule', {})
        phases = design_module.get('phases', [])
        phase = ClinicalTrialsParser.normalize_phase(phases)
        study_type = design_module.get('studyType', '')

        enrollment_info = design_module.get('enrollmentInfo', {})
        enrollment = enrollment_info.get('count')

        # Interventions
        interventions = []
        arms_module = protocol.get('armsInterventionsModule', {})
        for intervention in arms_module.get('interventions', []):
            interventions.append({
                'type': intervention.get('type', ''),
                'name': intervention.get('name', ''),
                'description': intervention.get('description', '')
            })

        # Outcomes
        outcomes_module = protocol.get('outcomesModule', {})
        primary_outcomes = [
            {
                'measure': o.get('measure', ''),
                'timeFrame': o.get('timeFrame', ''),
                'description': o.get('description', '')
            }
            for o in outcomes_module.get('primaryOutcomes', [])
        ]

        secondary_outcomes = [
            {
                'measure': o.get('measure', ''),
                'timeFrame': o.get('timeFrame', ''),
                'description': o.get('description', '')
            }
            for o in outcomes_module.get('secondaryOutcomes', [])
        ]

        # Conditions
        conditions_module = protocol.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', [])

        # Description
        desc_module = protocol.get('descriptionModule', {})
        brief_summary = desc_module.get('briefSummary', '')
        detailed_description = desc_module.get('detailedDescription', '')

        # Eligibility
        eligibility_module = protocol.get('eligibilityModule', {})
        eligibility_criteria = eligibility_module.get('eligibilityCriteria', '')
        min_age = eligibility_module.get('minimumAge', '')
        max_age = eligibility_module.get('maximumAge', '')

        # Contacts
        contacts_module = protocol.get('contactsLocationsModule', {})
        locations = []
        for loc in contacts_module.get('locations', []):
            locations.append({
                'facility': loc.get('facility', ''),
                'city': loc.get('city', ''),
                'state': loc.get('state', ''),
                'country': loc.get('country', '')
            })

        return {
            'nct_id': nct_id,
            'title': title,
            'official_title': official_title,
            'status': status,
            'phase': phase,
            'study_type': study_type,
            'enrollment': enrollment,
            'start_date': start_date,
            'completion_date': completion_date,
            'lead_sponsor': lead_sponsor,
            'collaborators': collaborators,
            'interventions': interventions,
            'primary_outcomes': primary_outcomes,
            'secondary_outcomes': secondary_outcomes,
            'conditions': conditions,
            'brief_summary': brief_summary,
            'detailed_description': detailed_description,
            'eligibility_criteria': eligibility_criteria,
            'min_age': min_age,
            'max_age': max_age,
            'locations': locations
        }

    @staticmethod
    def normalize_phase(phases: List[str]) -> Optional[str]:
        """
        Normalize phase list to readable string.

        Args:
            phases: List of phase identifiers

        Returns:
            Normalized phase string
        """
        if not phases:
            return None

        normalized = []
        for phase in phases:
            normalized.append(
                ClinicalTrialsParser.PHASE_MAP.get(phase, phase)
            )

        return '/'.join(normalized)

    @staticmethod
    def _extract_date(date_struct: Dict) -> Optional[str]:
        """Extract date from date struct."""
        if not date_struct:
            return None
        return date_struct.get('date')
