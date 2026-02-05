# Medical Safety RAG: Research Documentation

## Overview

Standard RAG systems are insufficient for medical contexts. This research explores retrieval systems that:
1. Know when they don't know
2. Flag potential dangers proactively
3. Refuse to hallucinate
4. Provide confidence scores for clinical decision support

## Hypothesis

A RAG system with explicit safety guardrails, confidence scoring, and medical-specific checks will produce more reliable outputs for bedside clinical decision support than standard RAG approaches.

## Methodology

### Phase 1: Safety Guardrails
- Confidence scoring based on retrieval quality
- Hallucination detection via claim verification
- Explicit uncertainty when evidence is insufficient
- Source quality weighting

### Phase 2: Medical-Specific Retrieval
- Drug interaction awareness
- Contraindication highlighting
- Dosing range verification
- Patient population sensitivity (pediatric, geriatric, pregnancy)

### Phase 3: Evaluation Framework
- Adversarial testing (attempt to elicit dangerous advice)
- False negative rate measurement
- Comparison against gold-standard references
- Clinician validation studies

## Safety-Critical Design Principles

1. **Fail Safe**: When uncertain, refuse to answer rather than guess
2. **Cite Everything**: Every claim must trace to a source
3. **Flag Dangers**: Proactively surface warnings, contraindications
4. **Know Limits**: Explicit about what the system cannot do
5. **Human in Loop**: Designed for decision *support*, not replacement

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Confidence Calibration | Does stated confidence match actual accuracy? | r² > 0.8 |
| Hallucination Rate | % of claims not supported by retrieved context | < 5% |
| False Negative Rate | % of missed critical warnings | < 1% |
| Refusal Appropriateness | Correct refusals when evidence insufficient | > 95% |

## Author

Matthew LaPinta, PharmD, BCEMP  
Lead ED Clinical Pharmacist, NYU Langone  
Founder, Ironsight Therapeutics

## Disclaimer

This is a research exploration. Not for clinical use without validation.
