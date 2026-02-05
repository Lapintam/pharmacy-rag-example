"""
Medical Safety RAG Research Module

Experimental features for improving safety and accuracy
of RAG systems in medical contexts.

Author: Matthew LaPinta, PharmD, BCEMP
"""

from .safety_guardrails import (
    assess_response_safety,
    SafetyAssessment,
    ConfidenceLevel,
    RetrievalQualityAnalyzer,
    HallucinationDetector,
    MedicalSafetyChecker,
    get_safe_response_wrapper,
)

from .drug_interaction_check import (
    check_drug_interactions,
    format_interaction_report,
    InteractionCheckResult,
    DrugInteraction,
    InteractionSeverity,
)

__all__ = [
    # Safety guardrails
    'assess_response_safety',
    'SafetyAssessment',
    'ConfidenceLevel',
    'RetrievalQualityAnalyzer',
    'HallucinationDetector',
    'MedicalSafetyChecker',
    'get_safe_response_wrapper',
    
    # Drug interactions
    'check_drug_interactions',
    'format_interaction_report',
    'InteractionCheckResult',
    'DrugInteraction',
    'InteractionSeverity',
]
