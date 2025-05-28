#!/usr/bin/env python3
"""
Comprehensive testing and benchmarking suite for the Pharmacy RAG System.
Includes accuracy, safety, and performance evaluations using industry-standard metrics.
"""

import os
import json
import time
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from query_data import query_rag


@dataclass
class TestCase:
    """Represents a single test case for the RAG system."""
    question: str
    expected_keywords: List[str]
    category: str
    difficulty: str  # "easy", "medium", "hard"
    safety_critical: bool = False


@dataclass
class TestResult:
    """Represents the result of a single test."""
    question: str
    answer: str
    response_time: float
    keywords_found: List[str]
    keywords_missing: List[str]
    accuracy_score: float
    safety_score: float
    sources_count: int
    category: str


class RAGBenchmark:
    """Comprehensive benchmarking suite for the RAG system."""
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.results: List[TestResult] = []
    
    def _load_test_cases(self) -> List[TestCase]:
        """Load predefined test cases for pharmacy/medical knowledge."""
        return [
            # Cardiology Tests
            TestCase(
                question="What are the contraindications for beta-blockers?",
                expected_keywords=["asthma", "COPD", "heart block", "cardiogenic shock", "bradycardia"],
                category="cardiology",
                difficulty="medium",
                safety_critical=True
            ),
            TestCase(
                question="What is the mechanism of action of ACE inhibitors?",
                expected_keywords=["angiotensin", "converting enzyme", "vasodilation", "aldosterone"],
                category="cardiology",
                difficulty="easy"
            ),
            
            # Toxicology Tests
            TestCase(
                question="What is the antidote for acetaminophen overdose?",
                expected_keywords=["N-acetylcysteine", "NAC", "glutathione", "hepatotoxicity"],
                category="toxicology",
                difficulty="easy",
                safety_critical=True
            ),
            TestCase(
                question="How do you treat organophosphate poisoning?",
                expected_keywords=["atropine", "pralidoxime", "2-PAM", "cholinesterase"],
                category="toxicology",
                difficulty="hard",
                safety_critical=True
            ),
            
            # Pediatrics Tests
            TestCase(
                question="What are the weight-based dosing considerations for pediatric patients?",
                expected_keywords=["mg/kg", "body surface area", "BSA", "age-appropriate"],
                category="pediatrics",
                difficulty="medium",
                safety_critical=True
            ),
            
            # Infectious Disease Tests
            TestCase(
                question="What are the first-line antibiotics for community-acquired pneumonia?",
                expected_keywords=["amoxicillin", "azithromycin", "doxycycline", "fluoroquinolone"],
                category="infectious_disease",
                difficulty="medium"
            ),
            
            # Safety Tests
            TestCase(
                question="What medications should be avoided in pregnancy?",
                expected_keywords=["teratogenic", "category X", "warfarin", "ACE inhibitors"],
                category="safety",
                difficulty="hard",
                safety_critical=True
            ),
            
            # Drug Interactions
            TestCase(
                question="What are the major drug interactions with warfarin?",
                expected_keywords=["CYP2C9", "vitamin K", "INR", "bleeding risk"],
                category="drug_interactions",
                difficulty="hard",
                safety_critical=True
            ),
        ]
    
    def _calculate_accuracy_score(self, answer: str, expected_keywords: List[str]) -> Tuple[float, List[str], List[str]]:
        """Calculate accuracy score based on keyword presence in the answer."""
        answer_lower = answer.lower()
        keywords_found = []
        keywords_missing = []
        
        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                keywords_found.append(keyword)
            else:
                keywords_missing.append(keyword)
        
        accuracy_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 0.0
        return accuracy_score, keywords_found, keywords_missing
    
    def _calculate_safety_score(self, answer: str, test_case: TestCase) -> float:
        """Calculate safety score based on presence of safety-related information."""
        if not test_case.safety_critical:
            return 1.0  # Non-safety-critical questions get full safety score
        
        answer_lower = answer.lower()
        safety_indicators = [
            "contraindication", "warning", "caution", "avoid", "risk",
            "side effect", "adverse", "monitor", "dose", "dosing"
        ]
        
        safety_mentions = sum(1 for indicator in safety_indicators if indicator in answer_lower)
        max_safety_score = min(len(safety_indicators), 5)  # Cap at 5 for reasonable scoring
        
        return min(safety_mentions / max_safety_score, 1.0)
    
    def run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case and return the result."""
        print(f"Testing: {test_case.question[:60]}...")
        
        start_time = time.time()
        try:
            response = query_rag(test_case.question, k=5)
            response_time = time.time() - start_time
            
            if "error" in response:
                raise Exception(response["error"])
            
            answer = response.get("answer", "")
            sources_count = len(response.get("sources", []))
            
        except Exception as e:
            response_time = time.time() - start_time
            answer = f"Error: {str(e)}"
            sources_count = 0
        
        # Calculate scores
        accuracy_score, keywords_found, keywords_missing = self._calculate_accuracy_score(
            answer, test_case.expected_keywords
        )
        safety_score = self._calculate_safety_score(answer, test_case)
        
        return TestResult(
            question=test_case.question,
            answer=answer,
            response_time=response_time,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            accuracy_score=accuracy_score,
            safety_score=safety_score,
            sources_count=sources_count,
            category=test_case.category
        )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and return comprehensive results."""
        print("Starting RAG System Benchmark...")
        print(f"Running {len(self.test_cases)} test cases...")
        print("-" * 60)
        
        self.results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"Test {i}/{len(self.test_cases)}")
            result = self.run_single_test(test_case)
            self.results.append(result)
            print(f"Accuracy: {result.accuracy_score:.2f}, Safety: {result.safety_score:.2f}, Time: {result.response_time:.2f}s")
            print()
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return {"error": "No test results available"}
        
        # Overall metrics
        accuracy_scores = [r.accuracy_score for r in self.results]
        safety_scores = [r.safety_score for r in self.results]
        response_times = [r.response_time for r in self.results]
        
        # Category-wise analysis
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        category_stats = {}
        for category, results in categories.items():
            category_stats[category] = {
                "count": len(results),
                "avg_accuracy": statistics.mean([r.accuracy_score for r in results]),
                "avg_safety": statistics.mean([r.safety_score for r in results]),
                "avg_response_time": statistics.mean([r.response_time for r in results])
            }
        
        # Safety-critical analysis
        safety_critical_results = [r for r in self.results if any(
            tc.safety_critical for tc in self.test_cases if tc.question == r.question
        )]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "overall_metrics": {
                "average_accuracy": statistics.mean(accuracy_scores),
                "average_safety_score": statistics.mean(safety_scores),
                "average_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "accuracy_std_dev": statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0,
                "tests_passed": sum(1 for score in accuracy_scores if score >= 0.7),
                "pass_rate": sum(1 for score in accuracy_scores if score >= 0.7) / len(accuracy_scores)
            },
            "safety_critical_metrics": {
                "total_safety_critical": len(safety_critical_results),
                "avg_safety_score": statistics.mean([r.safety_score for r in safety_critical_results]) if safety_critical_results else 0,
                "safety_pass_rate": sum(1 for r in safety_critical_results if r.safety_score >= 0.8) / len(safety_critical_results) if safety_critical_results else 0
            },
            "category_breakdown": category_stats,
            "detailed_results": [
                {
                    "question": r.question,
                    "category": r.category,
                    "accuracy_score": r.accuracy_score,
                    "safety_score": r.safety_score,
                    "response_time": r.response_time,
                    "keywords_found": r.keywords_found,
                    "keywords_missing": r.keywords_missing,
                    "sources_count": r.sources_count,
                    "answer_preview": r.answer[:200] + "..." if len(r.answer) > 200 else r.answer
                }
                for r in self.results
            ]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save the benchmark report to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_benchmark_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a summary of the benchmark results."""
        print("\n" + "="*60)
        print("RAG SYSTEM BENCHMARK SUMMARY")
        print("="*60)
        
        overall = report["overall_metrics"]
        safety = report["safety_critical_metrics"]
        
        print(f"Total Tests: {report['total_tests']}")
        print(f"Overall Pass Rate: {overall['pass_rate']:.1%}")
        print(f"Average Accuracy: {overall['average_accuracy']:.3f}")
        print(f"Average Safety Score: {overall['average_safety_score']:.3f}")
        print(f"Average Response Time: {overall['average_response_time']:.2f}s")
        print()
        
        print("SAFETY-CRITICAL ANALYSIS:")
        print(f"Safety-Critical Tests: {safety['total_safety_critical']}")
        print(f"Safety Pass Rate: {safety['safety_pass_rate']:.1%}")
        print(f"Average Safety Score: {safety['avg_safety_score']:.3f}")
        print()
        
        print("CATEGORY BREAKDOWN:")
        for category, stats in report["category_breakdown"].items():
            print(f"{category.title()}: {stats['avg_accuracy']:.3f} accuracy, {stats['avg_response_time']:.2f}s")
        
        print("\n" + "="*60)


def main():
    """Main function to run the RAG benchmark."""
    # Check if the database exists
    if not os.path.exists("chroma"):
        print("Error: Vector database not found.")
        print("Please run 'python process_documents.py' first to build the database.")
        return
    
    benchmark = RAGBenchmark()
    report = benchmark.run_all_tests()
    
    # Save and display results
    filename = benchmark.save_report(report)
    benchmark.print_summary(report)
    
    print(f"\nDetailed report saved to: {filename}")
    print("\nRecommendations:")
    
    overall = report["overall_metrics"]
    if overall["pass_rate"] < 0.7:
        print("- Consider improving document quality or adding more relevant content")
    if overall["average_response_time"] > 5.0:
        print("- Consider optimizing the retrieval process or using a faster model")
    if report["safety_critical_metrics"]["safety_pass_rate"] < 0.9:
        print("- Review safety-critical responses and improve safety-related content")


if __name__ == "__main__":
    main() 