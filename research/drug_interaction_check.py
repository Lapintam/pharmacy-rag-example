"""
Drug Interaction Checker for Medical RAG Systems

This module provides drug interaction awareness:
1. Multi-drug query detection
2. Known interaction flagging
3. Severity classification
4. Recommendation generation

Author: Matthew LaPinta, PharmD, BCEMP
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class InteractionSeverity(Enum):
    """Severity levels for drug interactions."""
    CONTRAINDICATED = "contraindicated"  # Do not use together
    MAJOR = "major"                       # May be life-threatening
    MODERATE = "moderate"                 # May require intervention
    MINOR = "minor"                       # Minimal clinical significance
    UNKNOWN = "unknown"                   # Not enough data


@dataclass
class DrugInteraction:
    """Container for drug interaction information."""
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    mechanism: str
    clinical_effect: str
    recommendation: str


@dataclass
class InteractionCheckResult:
    """Results from interaction checking."""
    drugs_detected: List[str]
    interactions_found: List[DrugInteraction]
    warnings: List[str]
    requires_review: bool


class DrugExtractor:
    """Extracts drug names from text using pattern matching."""
    
    # Common drug name suffixes (helps identify generic names)
    DRUG_SUFFIXES = [
        'olol', 'pril', 'sartan', 'statin', 'prazole', 'tidine',
        'cycline', 'mycin', 'cillin', 'floxacin', 'azole',
        'pam', 'lam', 'pine', 'done', 'ine', 'ide', 'ate',
        'mab', 'nib', 'tinib', 'zumab', 'ximab',
    ]
    
    # Common high-risk drugs (always flag these)
    HIGH_RISK_DRUGS = {
        'warfarin', 'heparin', 'enoxaparin', 'apixaban', 'rivaroxaban', 'dabigatran',
        'methotrexate', 'lithium', 'digoxin', 'phenytoin', 'carbamazepine',
        'theophylline', 'aminophylline', 'vancomycin', 'gentamicin', 'tobramycin',
        'insulin', 'metformin', 'sulfonylurea', 'glipizide', 'glyburide',
        'opioid', 'morphine', 'fentanyl', 'hydromorphone', 'oxycodone',
        'benzodiazepine', 'lorazepam', 'midazolam', 'diazepam',
        'potassium', 'magnesium', 'calcium', 'sodium',
    }
    
    # Common drug classes
    DRUG_CLASSES = {
        'beta-blocker': ['metoprolol', 'carvedilol', 'atenolol', 'propranolol', 'bisoprolol'],
        'ace-inhibitor': ['lisinopril', 'enalapril', 'ramipril', 'benazepril', 'captopril'],
        'arb': ['losartan', 'valsartan', 'irbesartan', 'olmesartan', 'candesartan'],
        'statin': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin'],
        'ppi': ['omeprazole', 'pantoprazole', 'esomeprazole', 'lansoprazole'],
        'ssri': ['sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram'],
        'nsaid': ['ibuprofen', 'naproxen', 'ketorolac', 'indomethacin', 'diclofenac'],
        'anticoagulant': ['warfarin', 'heparin', 'enoxaparin', 'apixaban', 'rivaroxaban'],
        'antiplatelet': ['aspirin', 'clopidogrel', 'prasugrel', 'ticagrelor'],
        'diuretic': ['furosemide', 'hydrochlorothiazide', 'spironolactone', 'bumetanide'],
        'opioid': ['morphine', 'fentanyl', 'hydromorphone', 'oxycodone', 'hydrocodone'],
        'fluoroquinolone': ['ciprofloxacin', 'levofloxacin', 'moxifloxacin'],
    }
    
    @classmethod
    def extract_drugs(cls, text: str) -> Set[str]:
        """Extract potential drug names from text."""
        drugs = set()
        text_lower = text.lower()
        
        # Check for high-risk drugs
        for drug in cls.HIGH_RISK_DRUGS:
            if drug in text_lower:
                drugs.add(drug)
        
        # Check drug classes and their members
        for drug_class, members in cls.DRUG_CLASSES.items():
            if drug_class.replace('-', ' ') in text_lower or drug_class in text_lower:
                drugs.add(drug_class)
            for member in members:
                if member in text_lower:
                    drugs.add(member)
        
        # Pattern match for potential drug names (capitalized words with drug suffixes)
        for suffix in cls.DRUG_SUFFIXES:
            pattern = rf'\b\w+{suffix}\b'
            matches = re.findall(pattern, text_lower)
            drugs.update(matches)
        
        return drugs
    
    @classmethod
    def get_drug_class(cls, drug: str) -> Optional[str]:
        """Get the drug class for a given drug."""
        drug_lower = drug.lower()
        
        for drug_class, members in cls.DRUG_CLASSES.items():
            if drug_lower in members or drug_lower == drug_class:
                return drug_class
        
        return None


class InteractionDatabase:
    """
    Known drug interactions database.
    
    In production, this would connect to a comprehensive database.
    This is a simplified version for demonstration.
    """
    
    # Known significant interactions (simplified)
    KNOWN_INTERACTIONS: Dict[Tuple[str, str], DrugInteraction] = {}
    
    @classmethod
    def _init_interactions(cls):
        """Initialize known interactions."""
        if cls.KNOWN_INTERACTIONS:
            return
        
        interactions = [
            # Anticoagulant interactions
            DrugInteraction(
                drug_a="warfarin", drug_b="nsaid",
                severity=InteractionSeverity.MAJOR,
                mechanism="NSAIDs inhibit platelet function and may cause GI bleeding",
                clinical_effect="Increased risk of bleeding",
                recommendation="Avoid combination if possible. If necessary, monitor closely for bleeding."
            ),
            DrugInteraction(
                drug_a="warfarin", drug_b="ssri",
                severity=InteractionSeverity.MODERATE,
                mechanism="SSRIs inhibit platelet aggregation",
                clinical_effect="Increased risk of bleeding",
                recommendation="Monitor for signs of bleeding. Consider PPI for GI protection."
            ),
            DrugInteraction(
                drug_a="anticoagulant", drug_b="antiplatelet",
                severity=InteractionSeverity.MAJOR,
                mechanism="Additive effects on hemostasis",
                clinical_effect="Significantly increased bleeding risk",
                recommendation="Use combination only when clearly indicated. Monitor closely."
            ),
            
            # QT prolongation combinations
            DrugInteraction(
                drug_a="fluoroquinolone", drug_b="ssri",
                severity=InteractionSeverity.MODERATE,
                mechanism="Both can prolong QT interval",
                clinical_effect="Increased risk of cardiac arrhythmias",
                recommendation="Monitor ECG. Avoid in patients with baseline QT prolongation."
            ),
            
            # ACE/ARB + Potassium
            DrugInteraction(
                drug_a="ace-inhibitor", drug_b="potassium",
                severity=InteractionSeverity.MODERATE,
                mechanism="ACE inhibitors reduce aldosterone, decreasing potassium excretion",
                clinical_effect="Risk of hyperkalemia",
                recommendation="Monitor potassium levels closely."
            ),
            DrugInteraction(
                drug_a="arb", drug_b="potassium",
                severity=InteractionSeverity.MODERATE,
                mechanism="ARBs reduce aldosterone, decreasing potassium excretion",
                clinical_effect="Risk of hyperkalemia",
                recommendation="Monitor potassium levels closely."
            ),
            
            # Serotonin syndrome risk
            DrugInteraction(
                drug_a="ssri", drug_b="opioid",
                severity=InteractionSeverity.MODERATE,
                mechanism="Some opioids (tramadol, fentanyl) have serotonergic activity",
                clinical_effect="Risk of serotonin syndrome",
                recommendation="Monitor for serotonin syndrome symptoms. Use lowest effective doses."
            ),
            
            # Statin interactions
            DrugInteraction(
                drug_a="simvastatin", drug_b="amiodarone",
                severity=InteractionSeverity.MAJOR,
                mechanism="Amiodarone inhibits CYP3A4, increasing statin levels",
                clinical_effect="Increased risk of rhabdomyolysis",
                recommendation="Do not exceed simvastatin 20mg daily with amiodarone."
            ),
            
            # Methotrexate interactions
            DrugInteraction(
                drug_a="methotrexate", drug_b="nsaid",
                severity=InteractionSeverity.MAJOR,
                mechanism="NSAIDs decrease methotrexate renal clearance",
                clinical_effect="Increased methotrexate toxicity",
                recommendation="Avoid NSAIDs with high-dose methotrexate. Monitor closely with low-dose."
            ),
            
            # Digoxin interactions
            DrugInteraction(
                drug_a="digoxin", drug_b="amiodarone",
                severity=InteractionSeverity.MAJOR,
                mechanism="Amiodarone increases digoxin levels",
                clinical_effect="Risk of digoxin toxicity",
                recommendation="Reduce digoxin dose by 50% when starting amiodarone."
            ),
        ]
        
        for interaction in interactions:
            key = tuple(sorted([interaction.drug_a, interaction.drug_b]))
            cls.KNOWN_INTERACTIONS[key] = interaction
    
    @classmethod
    def check_interaction(cls, drug_a: str, drug_b: str) -> Optional[DrugInteraction]:
        """Check if two drugs have a known interaction."""
        cls._init_interactions()
        
        drug_a_lower = drug_a.lower()
        drug_b_lower = drug_b.lower()
        
        # Direct lookup
        key = tuple(sorted([drug_a_lower, drug_b_lower]))
        if key in cls.KNOWN_INTERACTIONS:
            return cls.KNOWN_INTERACTIONS[key]
        
        # Check by drug class
        class_a = DrugExtractor.get_drug_class(drug_a_lower)
        class_b = DrugExtractor.get_drug_class(drug_b_lower)
        
        if class_a:
            key = tuple(sorted([class_a, drug_b_lower]))
            if key in cls.KNOWN_INTERACTIONS:
                return cls.KNOWN_INTERACTIONS[key]
        
        if class_b:
            key = tuple(sorted([drug_a_lower, class_b]))
            if key in cls.KNOWN_INTERACTIONS:
                return cls.KNOWN_INTERACTIONS[key]
        
        if class_a and class_b:
            key = tuple(sorted([class_a, class_b]))
            if key in cls.KNOWN_INTERACTIONS:
                return cls.KNOWN_INTERACTIONS[key]
        
        return None


def check_drug_interactions(text: str) -> InteractionCheckResult:
    """
    Check text for potential drug interactions.
    
    Args:
        text: Query or response text to analyze
        
    Returns:
        InteractionCheckResult with detected drugs and interactions
    """
    # Extract drugs from text
    drugs = DrugExtractor.extract_drugs(text)
    drugs_list = list(drugs)
    
    interactions = []
    warnings = []
    
    # Check pairwise interactions
    for i, drug_a in enumerate(drugs_list):
        for drug_b in drugs_list[i+1:]:
            interaction = InteractionDatabase.check_interaction(drug_a, drug_b)
            if interaction:
                interactions.append(interaction)
                
                if interaction.severity in [InteractionSeverity.CONTRAINDICATED, InteractionSeverity.MAJOR]:
                    warnings.append(
                        f"⚠️ {interaction.severity.value.upper()}: {drug_a} + {drug_b} - "
                        f"{interaction.clinical_effect}"
                    )
    
    # Check for high-risk drugs
    high_risk_found = drugs.intersection(DrugExtractor.HIGH_RISK_DRUGS)
    if high_risk_found:
        warnings.append(f"High-risk medication(s) detected: {', '.join(high_risk_found)}")
    
    # Determine if requires review
    requires_review = (
        any(i.severity in [InteractionSeverity.CONTRAINDICATED, InteractionSeverity.MAJOR] 
            for i in interactions) or
        len(high_risk_found) > 1 or
        len(drugs) > 4  # Polypharmacy concern
    )
    
    return InteractionCheckResult(
        drugs_detected=drugs_list,
        interactions_found=interactions,
        warnings=warnings,
        requires_review=requires_review
    )


def format_interaction_report(result: InteractionCheckResult) -> str:
    """Format interaction check results for display."""
    lines = []
    
    lines.append(f"Drugs detected: {', '.join(result.drugs_detected) if result.drugs_detected else 'None'}")
    lines.append("")
    
    if result.interactions_found:
        lines.append("Interactions Found:")
        for interaction in result.interactions_found:
            lines.append(f"  [{interaction.severity.value.upper()}] {interaction.drug_a} + {interaction.drug_b}")
            lines.append(f"    Effect: {interaction.clinical_effect}")
            lines.append(f"    Recommendation: {interaction.recommendation}")
            lines.append("")
    else:
        lines.append("No known interactions found.")
        lines.append("")
    
    if result.warnings:
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  {warning}")
        lines.append("")
    
    if result.requires_review:
        lines.append("⚠️ PHARMACIST REVIEW RECOMMENDED")
    
    return '\n'.join(lines)
