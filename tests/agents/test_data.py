"""Sample clinical trial data for testing multi-agent workflows."""

# Sample 1: Phase 2 CAR-T Therapy Trial
CART_LYMPHOMA_TRIAL = {
    "nct_id": "NCT04567890",
    "title": "Phase 2 Study of Novel CD19/CD22 Dual-Targeting CAR-T in Relapsed/Refractory B-Cell Lymphoma",
    "phase": "Phase II",
    "indication": "Relapsed/refractory B-cell lymphoma",
    "intervention": "CD19/CD22 dual-targeting CAR-T cell therapy",
    "sample_size": 60,
    "primary_outcome": "Overall response rate (ORR) at 3 months",
    "secondary_outcomes": "Duration of response, progression-free survival, safety and tolerability, CAR-T expansion kinetics",
    "sponsor": "Boston Children's Hospital",
    "status": "Recruiting",
    "description": """
    This is a Phase 2, single-arm, open-label study evaluating a novel dual-targeting
    CAR-T therapy in patients with relapsed/refractory B-cell lymphoma who have failed
    at least 2 prior lines of therapy. The CAR-T product targets both CD19 and CD22
    antigens to prevent antigen escape, a major limitation of current single-target
    CAR-T therapies. The study will enroll 60 patients across 8 sites and assess efficacy,
    safety, and CAR-T expansion/persistence.
    """
}

# Sample 2: Phase 3 Pediatric Asthma Trial
PEDIATRIC_ASTHMA_TRIAL = {
    "nct_id": "NCT05123456",
    "title": "Phase 3 Randomized Trial of Monoclonal Antibody XYZ-301 in Severe Pediatric Asthma",
    "phase": "Phase III",
    "indication": "Severe asthma, pediatric (ages 6-17)",
    "intervention": "XYZ-301 monoclonal antibody (IL-5 receptor alpha antagonist)",
    "sample_size": 450,
    "primary_outcome": "Rate of asthma exacerbations over 52 weeks",
    "secondary_outcomes": "FEV1 improvement, asthma control questionnaire scores, quality of life, steroid-sparing effect",
    "sponsor": "Boston Children's Hospital / Pharmaceutical Partner",
    "status": "Active, not recruiting",
    "description": """
    This is a Phase 3, randomized, double-blind, placebo-controlled trial of XYZ-301,
    a novel monoclonal antibody targeting the IL-5 receptor alpha, in children with
    severe asthma. The trial will compare XYZ-301 administered subcutaneously every 8
    weeks versus placebo, on top of standard of care inhaled corticosteroids and long-acting
    beta agonists. The study aims to demonstrate superiority in reducing asthma exacerbations
    compared to existing therapies like dupilumab and mepolizumab.
    """
}

# Sample 3: Early-stage Gene Therapy Trial
GENE_THERAPY_TRIAL = {
    "nct_id": "NCT04789012",
    "title": "Phase 1/2 Safety and Efficacy Study of AAV9-Based Gene Therapy for Niemann-Pick Disease Type C",
    "phase": "Phase I/II",
    "indication": "Niemann-Pick Disease Type C (ultra-rare lysosomal storage disorder)",
    "intervention": "AAV9-NPC1 gene therapy (single administration)",
    "sample_size": 12,
    "primary_outcome": "Safety and tolerability at 6 months",
    "secondary_outcomes": "NPC1 protein expression, cholesterol storage markers, neurocognitive function, disease progression",
    "sponsor": "Boston Children's Hospital",
    "status": "Recruiting",
    "description": """
    This is a first-in-human, Phase 1/2, open-label, dose-escalation study of AAV9-mediated
    gene therapy for Niemann-Pick Disease Type C, an ultra-rare, fatal neurodegenerative
    disorder with no approved therapies. The study will enroll 12 pediatric patients in 3
    dose cohorts and assess safety, NPC1 protein expression, and preliminary efficacy signals.
    This represents a novel therapeutic approach for an orphan disease with significant unmet need.
    """
}

# Sample 4: Medical Device Trial
DEVICE_TRIAL = {
    "nct_id": "NCT04654321",
    "title": "Pivotal Trial of Wearable Seizure Detection Device in Pediatric Epilepsy",
    "phase": "N/A (Device)",
    "indication": "Pediatric epilepsy (ages 2-18)",
    "intervention": "NeuroGuard wearable seizure detection and alert system",
    "sample_size": 200,
    "primary_outcome": "Sensitivity and specificity of seizure detection vs. video-EEG gold standard",
    "secondary_outcomes": "False alarm rate, battery life, user satisfaction, time to caregiver notification",
    "sponsor": "Boston Children's Hospital / MedTech Startup",
    "status": "Enrolling by invitation",
    "description": """
    This is a pivotal clinical trial for FDA approval of a novel wearable device that
    continuously monitors for seizures in children with epilepsy and alerts caregivers
    in real-time. The device uses accelerometry, heart rate, and EDA sensors with a
    machine learning algorithm. The study will compare device performance against video-EEG
    monitoring (gold standard) in both hospital and home settings. Market opportunity includes
    1.2M children with epilepsy in the US, with high parental demand for monitoring solutions.
    """
}

# Sample 5: Combination Therapy Trial
COMBINATION_TRIAL = {
    "nct_id": "NCT04999888",
    "title": "Phase 2 Study of Checkpoint Inhibitor Plus CAR-T Therapy in Pediatric Solid Tumors",
    "phase": "Phase II",
    "indication": "Relapsed/refractory pediatric solid tumors (neuroblastoma, sarcoma)",
    "intervention": "Anti-PD1 antibody + GD2-targeting CAR-T cells",
    "sample_size": 30,
    "primary_outcome": "Best overall response rate",
    "secondary_outcomes": "Duration of response, progression-free survival, CAR-T expansion with vs. without PD-1 blockade",
    "sponsor": "Boston Children's Hospital",
    "status": "Not yet recruiting",
    "description": """
    This Phase 2 trial evaluates the combination of checkpoint inhibition (anti-PD1) with
    GD2-targeting CAR-T cells in pediatric solid tumors. Preclinical data suggests PD-1
    blockade enhances CAR-T expansion and efficacy in the immunosuppressive tumor
    microenvironment. This represents a novel combination strategy to overcome CAR-T
    resistance in solid tumors, addressing a major unmet need in pediatric oncology.
    """
}

# Collection of all samples
ALL_SAMPLE_TRIALS = [
    CART_LYMPHOMA_TRIAL,
    PEDIATRIC_ASTHMA_TRIAL,
    GENE_THERAPY_TRIAL,
    DEVICE_TRIAL,
    COMBINATION_TRIAL
]

# Expected evaluation outcomes (for testing)
EXPECTED_EVALUATIONS = {
    "NCT04567890": {  # CAR-T
        "expected_score_range": (60, 75),
        "expected_recommendation": "MEDIUM",
        "key_factors": [
            "Competitive CAR-T market",
            "Dual-targeting innovation",
            "Small patient population"
        ]
    },
    "NCT05123456": {  # Pediatric asthma
        "expected_score_range": (55, 70),
        "expected_recommendation": "MEDIUM",
        "key_factors": [
            "Large market ($2-3B)",
            "Strong competitor (dupilumab)",
            "Late entry timing"
        ]
    },
    "NCT04789012": {  # Gene therapy
        "expected_score_range": (65, 80),
        "expected_recommendation": "MEDIUM",
        "key_factors": [
            "High unmet need",
            "Orphan designation likely",
            "Small market, early stage"
        ]
    },
    "NCT04654321": {  # Device
        "expected_score_range": (70, 85),
        "expected_recommendation": "HIGH",
        "key_factors": [
            "Large addressable market",
            "High parental demand",
            "Clear regulatory pathway"
        ]
    },
    "NCT04999888": {  # Combination
        "expected_score_range": (55, 70),
        "expected_recommendation": "MEDIUM",
        "key_factors": [
            "Novel combination approach",
            "Small market (pediatric solid tumors)",
            "Early stage, needs validation"
        ]
    }
}
