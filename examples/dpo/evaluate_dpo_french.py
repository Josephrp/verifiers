#!/usr/bin/env python3
"""
Evaluation Script for DPO French Reasoning Models

This script evaluates trained models on French reasoning tasks,
validating thinking generation, translation quality, and preference alignment.

Usage:
    python examples/dpo/evaluate_dpo_french.py --model your-trained-model --baseline-model base-model
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import re

import verifiers as vf
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.metrics import accuracy_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DPO French Reasoning Models")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--baseline_model", type=str, help="Path to baseline model for comparison")
    parser.add_argument("--env_id", type=str, default="vf-dpo-french-reasoning", help="Environment ID")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens")
    parser.add_argument("--output_file", type=str, help="Output file for detailed results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


class FrenchReasoningEvaluator:
    """Evaluator for French reasoning models with comprehensive metrics"""

    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='fr')

    def is_french(self, text: str) -> bool:
        """Check if text is in French"""
        try:
            return detect(text) == 'fr'
        except:
            return False

    def extract_thinking_and_answer(self, completion: str) -> Dict[str, str]:
        """Extract thinking trace and final answer from completion"""
        thinking_patterns = [
            r'(?i)Raisonnement:(.*?)(?=Réponse:|$)',
            r'(?i)Pensons étape par étape:(.*?)(?=Réponse:|$)',
            r'(?i)Raisonnons:(.*?)(?=Réponse:|$)'
        ]

        thinking = ""
        for pattern in thinking_patterns:
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                break

        # Extract answer (everything after the last thinking section)
        answer = completion
        if thinking:
            answer = completion.split(thinking)[-1].strip()
            # Remove common separators
            for sep in ['Réponse:', 'Answer:', 'Final answer:']:
                if answer.startswith(sep):
                    answer = answer[len(sep):].strip()

        return {
            'thinking': thinking,
            'answer': answer,
            'full_response': completion
        }

    def evaluate_french_quality(self, text: str) -> Dict[str, float]:
        """Evaluate French language quality"""
        if not text.strip():
            return {'french_score': 0.0, 'structure_score': 0.0}

        # French language detection
        is_french = self.is_french(text)
        french_score = 1.0 if is_french else 0.0

        # Structure indicators
        structure_indicators = [
            'raisonnons', 'réfléchissons', 'étape', 'premièrement',
            'ensuite', 'donc', 'par conséquent', 'finalement'
        ]

        structure_score = 0.0
        text_lower = text.lower()
        for indicator in structure_indicators:
            if indicator in text_lower:
                structure_score += 0.1
        structure_score = min(1.0, structure_score)

        return {
            'french_score': french_score,
            'structure_score': structure_score
        }

    def evaluate_preference_alignment(self, response: str, accepted: str, rejected: str) -> Dict[str, float]:
        """Evaluate how well response aligns with preferences"""
        if not response or not accepted or not rejected:
            return {'preference_score': 0.0, 'accepted_similarity': 0.0, 'rejected_similarity': 0.0}

        response_lower = response.lower()
        accepted_lower = accepted.lower()
        rejected_lower = rejected.lower()

        # Simple keyword-based similarity
        accepted_keywords = set(accepted_lower.split())
        rejected_keywords = set(rejected_lower.split())
        response_keywords = set(response_lower.split())

        accepted_similarity = len(accepted_keywords.intersection(response_keywords)) / max(len(accepted_keywords), 1)
        rejected_similarity = len(rejected_keywords.intersection(response_keywords)) / max(len(rejected_keywords), 1)

        # Preference score: higher when similar to accepted, lower when similar to rejected
        preference_score = max(0.0, accepted_similarity - rejected_similarity)

        return {
            'preference_score': preference_score,
            'accepted_similarity': accepted_similarity,
            'rejected_similarity': rejected_similarity
        }

    def evaluate_completeness(self, response: str, expected_answer: str) -> Dict[str, float]:
        """Evaluate response completeness and correctness"""
        correctness = 1.0 if expected_answer.lower().strip() in response.lower() else 0.0

        # Length-based completeness (responses should be substantial)
        word_count = len(response.split())
        length_score = min(1.0, word_count / 50.0)  # Expect at least 50 words

        return {
            'correctness': correctness,
            'length_score': length_score,
            'completeness_score': (correctness + length_score) / 2.0
        }


def evaluate_model(model_path: str, env: vf.MultiTurnEnv, evaluator: FrenchReasoningEvaluator,
                  num_examples: int = 100, max_new_tokens: int = 256, model_name: str = "Model") -> Dict[str, Any]:
    """Evaluate a single model"""

    print(f"Loading {model_name}: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Evaluating {model_name} on {num_examples} examples...")

    # Run evaluation
    results = env.evaluate(
        model=model,
        tokenizer=tokenizer,
        num_examples=num_examples,
        max_new_tokens=max_new_tokens
    )

    # Detailed analysis
    detailed_results = []
    metrics = {
        'french_scores': [],
        'structure_scores': [],
        'preference_scores': [],
        'correctness_scores': [],
        'completeness_scores': []
    }

    for i, result in enumerate(results.data):
        prompt = result.get('prompt', '')
        completion = result.get('completion', '')
        accepted = result.get('accepted', '')
        rejected = result.get('rejected', '')
        expected_answer = result.get('original_answer', result.get('answer', ''))

        # Extract thinking and answer
        parsed = evaluator.extract_thinking_and_answer(completion)

        # Evaluate components
        french_metrics = evaluator.evaluate_french_quality(parsed['thinking'])
        preference_metrics = evaluator.evaluate_preference_alignment(
            parsed['full_response'], accepted, rejected
        )
        completeness_metrics = evaluator.evaluate_completeness(
            parsed['full_response'], expected_answer
        )

        # Store metrics
        for key in ['french_score', 'structure_score']:
            metrics[key + 's'].append(french_metrics[key])
        for key in ['preference_score']:
            metrics[key + 's'].append(preference_metrics[key])
        for key in ['correctness', 'completeness_score']:
            metrics[key + '_scores'].append(completeness_metrics[key])

        # Store detailed result
        detailed_results.append({
            'index': i,
            'prompt': prompt,
            'thinking': parsed['thinking'],
            'response': parsed['full_response'],
            'accepted': accepted,
            'rejected': rejected,
            'expected_answer': expected_answer,
            'french_metrics': french_metrics,
            'preference_metrics': preference_metrics,
            'completeness_metrics': completeness_metrics,
            'reward': result.get('reward', 0.0)
        })

    # Compute aggregate metrics
    aggregate_metrics = {}
    for key, values in metrics.items():
        if values:
            aggregate_metrics[f'avg_{key}'] = sum(values) / len(values)
            aggregate_metrics[f'min_{key}'] = min(values)
            aggregate_metrics[f'max_{key}'] = max(values)
        else:
            aggregate_metrics[f'avg_{key}'] = 0.0

    aggregate_metrics['num_examples'] = len(detailed_results)

    return {
        'model_name': model_name,
        'aggregate_metrics': aggregate_metrics,
        'detailed_results': detailed_results
    }


def compare_models(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare multiple models"""
    if len(model_results) < 2:
        return {}

    comparison = {}
    base_results = model_results[0]

    for result in model_results[1:]:
        model_name = result['model_name']
        comparison[model_name] = {}

        for metric in base_results['aggregate_metrics']:
            if metric.startswith('avg_'):
                base_value = base_results['aggregate_metrics'][metric]
                current_value = result['aggregate_metrics'][metric]
                improvement = current_value - base_value
                comparison[model_name][metric + '_improvement'] = improvement
                comparison[model_name][metric + '_percent_change'] = (
                    (improvement / base_value * 100) if base_value != 0 else 0
                )

    return comparison


def main():
    args = parse_args()

    # Load environment
    print(f"Loading environment: {args.env_id}")
    env = vf.load_environment(
        env_id=args.env_id,
        num_eval_examples=args.num_examples
    )

    # Initialize evaluator
    evaluator = FrenchReasoningEvaluator()

    # Evaluate models
    all_results = []

    # Evaluate main model
    main_results = evaluate_model(
        args.model, env, evaluator,
        args.num_examples, args.max_new_tokens,
        "Trained Model"
    )
    all_results.append(main_results)

    # Evaluate baseline if provided
    if args.baseline_model:
        baseline_results = evaluate_model(
            args.baseline_model, env, evaluator,
            args.num_examples, args.max_new_tokens,
            "Baseline Model"
        )
        all_results.append(baseline_results)

    # Compare models
    if len(all_results) > 1:
        comparison = compare_models(all_results)
        print("\\n=== MODEL COMPARISON ===")
        for model_name, metrics in comparison.items():
            print(f"\\n{model_name}:")
            for metric, value in metrics.items():
                if '_improvement' in metric:
                    print(f"  {metric}: {value:+.3f}")

    # Print detailed results
    print("\\n=== DETAILED RESULTS ===")
    for result in all_results:
        print(f"\\n{result['model_name']} Results:")
        agg = result['aggregate_metrics']
        print(f"  Average French Score: {agg.get('avg_french_scores', 0):.3f}")
        print(f"  Average Structure Score: {agg.get('avg_structure_scores', 0):.3f}")
        print(f"  Average Preference Score: {agg.get('avg_preference_scores', 0):.3f}")
        print(f"  Average Correctness: {agg.get('avg_correctness_scores', 0):.3f}")
        print(f"  Average Completeness: {agg.get('avg_completeness_scores', 0):.3f}")

        if args.verbose:
            print("\\n  Sample Results:")
            for detail in result['detailed_results'][:3]:
                print(f"    Example {detail['index'] + 1}:")
                print(f"      French Score: {detail['french_metrics']['french_score']:.2f}")
                print(f"      Preference Score: {detail['preference_metrics']['preference_score']:.2f}")
                print(f"      Correctness: {detail['completeness_metrics']['correctness']:.2f}")

    # Save detailed results if requested
    if args.output_file:
        output_data = {
            'evaluation_config': vars(args),
            'results': all_results
        }

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\\nDetailed results saved to: {args.output_file}")

    print("\\nEvaluation completed!")


if __name__ == "__main__":
    main()
