"""
Safety Guardrails for Medical RAG Systems

This module implements safety-critical features for medical retrieval:
1. Confidence scoring based on retrieval quality
2. Hallucination detection via claim verification
3. Uncertainty quantification
4. Source quality assessment

Author: Matthew LaPinta, PharmD, BCEMP
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for clinical decision support."""
    HIGH = "high"           # Strong evidence, multiple corroborating sources
    MODERATE = "moderate"   # Good evidence, some uncertainty
    LOW = "low"             # Limited evidence, use caution
    INSUFFICIENT = "insufficient"  # Cannot answer reliably - refuse


@dataclass
class SafetyAssessment:
    """Container for safety assessment results."""
    confidence_level: ConfidenceLevel
    confidence_score: float  # 0.0 - 1.0
    hallucination_risk: float  # 0.0 - 1.0
    source_quality_score: float  # 0.0 - 1.0
    warnings: List[str]
    should_defer_to_human: bool
    reasoning: str


class RetrievalQualityAnalyzer:
    """Analyzes quality of retrieved documents for confidence scoring."""
    
    # Similarity score thresholds (ChromaDB returns distance, lower = better)
    EXCELLENT_THRESHOLD = 0.3
    GOOD_THRESHOLD = 0.5
    FAIR_THRESHOLD = 0.7
    
    # Minimum chunks needed for different confidence levels
    MIN_CHUNKS_HIGH = 3
    MIN_CHUNKS_MODERATE = 2
    MIN_CHUNKS_LOW = 1
    
    @classmethod
    def calculate_retrieval_confidence(
        cls,
        similarity_scores: List[float],
        num_chunks: int
    ) -> Tuple[float, str]:
        """
        Calculate confidence based on retrieval quality.
        
        Args:
            similarity_scores: List of similarity scores from retrieval
            num_chunks: Number of chunks retrieved
            
        Returns:
            Tuple of (confidence_score, reasoning)
        """
        if not similarity_scores:
            return 0.0, "No relevant documents found"
        
        # Average similarity (convert distance to similarity if needed)
        avg_score = sum(similarity_scores) / len(similarity_scores)
        best_score = min(similarity_scores)  # Lower distance = better match
        
        # Calculate base confidence from similarity
        if best_score <= cls.EXCELLENT_THRESHOLD:
            base_confidence = 0.9
            quality = "excellent"
        elif best_score <= cls.GOOD_THRESHOLD:
            base_confidence = 0.7
            quality = "good"
        elif best_score <= cls.FAIR_THRESHOLD:
            base_confidence = 0.5
            quality = "fair"
        else:
            base_confidence = 0.3
            quality = "poor"
        
        # Adjust for number of corroborating sources
        if num_chunks >= cls.MIN_CHUNKS_HIGH:
            source_bonus = 0.1
        elif num_chunks >= cls.MIN_CHUNKS_MODERATE:
            source_bonus = 0.05
        else:
            source_bonus = 0.0
        
        final_confidence = min(1.0, base_confidence + source_bonus)
        
        reasoning = (
            f"Retrieval quality: {quality} (best match: {best_score:.3f}). "
            f"Found {num_chunks} relevant chunks. "
            f"Confidence: {final_confidence:.2f}"
        )
        
        return final_confidence, reasoning


class HallucinationDetector:
    """Detects potential hallucinations by verifying claims against sources."""
    
    # Patterns that often indicate specific claims requiring verification
    CLAIM_PATTERNS = [
        r'\d+\s*(mg|mcg|g|ml|mL|units?|%)',  # Dosing claims
        r'(always|never|must|should not|contraindicated)',  # Absolute statements
        r'(first-line|second-line|drug of choice)',  # Treatment recommendations
        r'(within \d+|every \d+|q\d+h)',  # Timing claims
        r'(black box|boxed warning|FDA)',  # Regulatory claims
    ]
    
    @classmethod
    def extract_verifiable_claims(cls, response_text: str) -> List[str]:
        """Extract claims from response that should be verified."""
        claims = []
        sentences = response_text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            for pattern in cls.CLAIM_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(sentence)
                    break
        
        return claims
    
    @classmethod
    def verify_claim_against_context(
        cls,
        claim: str,
        context_chunks: List[str]
    ) -> Tuple[bool, float]:
        """
        Verify if a claim is supported by the retrieved context.
        
        Args:
            claim: The claim to verify
            context_chunks: List of retrieved text chunks
            
        Returns:
            Tuple of (is_supported, confidence)
        """
        claim_lower = claim.lower()
        combined_context = ' '.join(context_chunks).lower()
        
        # Extract key terms from claim
        # Remove common words and focus on medical terms, numbers
        key_terms = []
        
        # Look for drug names (typically capitalized or specific patterns)
        drug_pattern = r'\b[A-Z][a-z]+(?:in|ol|am|ide|one|ate|ine)\b'
        drugs = re.findall(drug_pattern, claim)
        key_terms.extend([d.lower() for d in drugs])
        
        # Look for numbers with units
        numbers = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|units?|%|hours?|days?)', claim)
        key_terms.extend([n.lower() for n in numbers])
        
        # Look for medical terms
        medical_terms = re.findall(r'\b(?:dose|dosage|contraindicated|indicated|first-line|warning|adverse|effect)\b', claim_lower)
        key_terms.extend(medical_terms)
        
        if not key_terms:
            # Can't verify - assume moderate risk
            return True, 0.5
        
        # Check how many key terms appear in context
        found_terms = sum(1 for term in key_terms if term in combined_context)
        coverage = found_terms / len(key_terms) if key_terms else 0
        
        is_supported = coverage >= 0.5
        confidence = coverage
        
        return is_supported, confidence
    
    @classmethod
    def assess_hallucination_risk(
        cls,
        response_text: str,
        context_chunks: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Assess overall hallucination risk for a response.
        
        Returns:
            Tuple of (risk_score, list of unverified claims)
        """
        claims = cls.extract_verifiable_claims(response_text)
        
        if not claims:
            return 0.2, []  # Low risk if no specific claims
        
        unverified_claims = []
        total_confidence = 0.0
        
        for claim in claims:
            is_supported, confidence = cls.verify_claim_against_context(
                claim, context_chunks
            )
            total_confidence += confidence
            
            if not is_supported:
                unverified_claims.append(claim)
        
        avg_confidence = total_confidence / len(claims) if claims else 0
        risk_score = 1.0 - avg_confidence
        
        return risk_score, unverified_claims


class MedicalSafetyChecker:
    """Checks for medical safety concerns in queries and responses."""
    
    # High-risk query patterns that require extra caution
    HIGH_RISK_PATTERNS = [
        (r'\b(overdose|toxic|poison|suicid|lethal)\b', "toxicology_alert"),
        (r'\b(pregnan|fetus|fetal|teratogen|breastfeed|lactat)\b', "pregnancy_alert"),
        (r'\b(pediatric|child|infant|neonat|newborn)\b', "pediatric_alert"),
        (r'\b(geriatric|elderly|renal impair|hepatic impair)\b', "special_population_alert"),
        (r'\b(allerg|anaphyla|hypersensitiv)\b', "allergy_alert"),
        (r'\b(interact|contraindic|drug.drug|combination)\b', "interaction_alert"),
    ]
    
    # Keywords that should trigger warnings in responses
    WARNING_KEYWORDS = [
        "black box warning",
        "boxed warning", 
        "contraindicated",
        "do not use",
        "fatal",
        "life-threatening",
        "serious adverse",
        "discontinue immediately",
    ]
    
    @classmethod
    def check_query_risk(cls, query: str) -> List[str]:
        """Identify risk factors in the query."""
        alerts = []
        query_lower = query.lower()
        
        for pattern, alert_type in cls.HIGH_RISK_PATTERNS:
            if re.search(pattern, query_lower):
                alerts.append(alert_type)
        
        return alerts
    
    @classmethod
    def check_response_warnings(
        cls,
        response: str,
        context_chunks: List[str]
    ) -> List[str]:
        """Check if response should include warnings from context."""
        warnings = []
        combined_context = ' '.join(context_chunks).lower()
        response_lower = response.lower()
        
        for keyword in cls.WARNING_KEYWORDS:
            # If warning exists in context but not in response, flag it
            if keyword in combined_context and keyword not in response_lower:
                warnings.append(
                    f"Context contains '{keyword}' but response may not adequately address it"
                )
        
        return warnings


def assess_response_safety(
    query: str,
    response: str,
    context_chunks: List[str],
    similarity_scores: List[float]
) -> SafetyAssessment:
    """
    Comprehensive safety assessment for a RAG response.
    
    Args:
        query: Original user query
        response: Generated response
        context_chunks: Retrieved context chunks
        similarity_scores: Similarity scores from retrieval
        
    Returns:
        SafetyAssessment with confidence, risks, and recommendations
    """
    warnings = []
    
    # 1. Assess retrieval quality
    retrieval_confidence, retrieval_reasoning = (
        RetrievalQualityAnalyzer.calculate_retrieval_confidence(
            similarity_scores, len(context_chunks)
        )
    )
    
    # 2. Check for hallucination risk
    hallucination_risk, unverified_claims = (
        HallucinationDetector.assess_hallucination_risk(response, context_chunks)
    )
    
    if unverified_claims:
        warnings.append(
            f"Potentially unverified claims: {'; '.join(unverified_claims[:3])}"
        )
    
    # 3. Check query risk factors
    query_risks = MedicalSafetyChecker.check_query_risk(query)
    for risk in query_risks:
        warnings.append(f"Query involves {risk.replace('_', ' ')}")
    
    # 4. Check for missing warnings
    missing_warnings = MedicalSafetyChecker.check_response_warnings(
        response, context_chunks
    )
    warnings.extend(missing_warnings)
    
    # 5. Calculate overall confidence
    # Penalize for hallucination risk and missing warnings
    confidence_score = retrieval_confidence * (1 - hallucination_risk * 0.5)
    
    if missing_warnings:
        confidence_score *= 0.8
    
    if query_risks:
        # High-risk queries need higher evidence threshold
        confidence_score *= 0.9
    
    # 6. Determine confidence level
    if confidence_score >= 0.75:
        confidence_level = ConfidenceLevel.HIGH
    elif confidence_score >= 0.5:
        confidence_level = ConfidenceLevel.MODERATE
    elif confidence_score >= 0.25:
        confidence_level = ConfidenceLevel.LOW
    else:
        confidence_level = ConfidenceLevel.INSUFFICIENT
    
    # 7. Determine if should defer to human
    should_defer = (
        confidence_level == ConfidenceLevel.INSUFFICIENT or
        hallucination_risk > 0.6 or
        len(warnings) > 3 or
        any('toxicology' in w or 'fatal' in w.lower() for w in warnings)
    )
    
    # 8. Build reasoning
    reasoning = (
        f"{retrieval_reasoning} "
        f"Hallucination risk: {hallucination_risk:.2f}. "
        f"Query risks: {query_risks if query_risks else 'none'}. "
        f"Warnings: {len(warnings)}."
    )
    
    return SafetyAssessment(
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        hallucination_risk=hallucination_risk,
        source_quality_score=retrieval_confidence,
        warnings=warnings,
        should_defer_to_human=should_defer,
        reasoning=reasoning
    )


# Convenience function for integration
def get_safe_response_wrapper(query_func):
    """
    Decorator to wrap query functions with safety assessment.
    
    Usage:
        @get_safe_response_wrapper
        def my_query_function(query, ...):
            ...
    """
    def wrapper(query: str, *args, **kwargs):
        result = query_func(query, *args, **kwargs)
        
        # Extract necessary info from result
        response = result.get('answer', '')
        sources = result.get('sources', [])
        scores = result.get('scores', [])
        chunks = [s.get('content', '') for s in sources]
        
        # Run safety assessment
        assessment = assess_response_safety(query, response, chunks, scores)
        
        # Add safety info to result
        result['safety'] = {
            'confidence_level': assessment.confidence_level.value,
            'confidence_score': assessment.confidence_score,
            'hallucination_risk': assessment.hallucination_risk,
            'warnings': assessment.warnings,
            'should_defer_to_human': assessment.should_defer_to_human,
            'reasoning': assessment.reasoning
        }
        
        return result
    
    return wrapper
