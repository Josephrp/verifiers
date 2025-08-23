#!/usr/bin/env python3
"""
Example usage of the DeepConf environment for confidence-aware LLM reasoning evaluation.

This script demonstrates how to:
1. Load and configure the DeepConf environment
2. Run evaluations with confidence analysis
3. Analyze confidence correlations
4. Simulate DeepConf filtering
"""

import verifiers as vf
from environments.deepconf.deepconf import (
    analyze_confidence_correlation,
    simulate_deepconf_filtering,
    confidence_weighted_majority_vote,
    filter_traces_by_confidence
)

def main():
    """Demonstrate DeepConf environment usage."""

    # Load DeepConf environment with AIME 2025 dataset
    print("Loading DeepConf environment...")
    env = vf.load_environment(
        env_id="deepconf",
        dataset_name="aime2025",
        confidence_metric="avg_trace_confidence",
        confidence_threshold=0.5,
        num_traces=32,
        num_eval_examples=5  # Small example for demo
    )

    print("Environment loaded successfully!")
    print(f"Dataset: {env.dataset}")
    print(f"Eval dataset size: {len(env.eval_dataset)}")

    # Example of running evaluation (requires OpenAI-compatible client)
    """
    # Setup client
    client = OpenAI(api_key="your-api-key")

    # Run evaluation
    print("Running evaluation...")
    results = env.evaluate(
        client=client,
        model="gpt-4.1-mini",
        num_examples=5,
        rollouts_per_example=1
    )

    # Analyze confidence correlation
    print("Analyzing confidence correlation...")
    confidence_analysis = analyze_confidence_correlation(results)
    print(f"Confidence analysis: {confidence_analysis}")

    # Simulate DeepConf filtering
    print("Simulating DeepConf filtering...")
    filtering_results = simulate_deepconf_filtering(
        results=results,
        confidence_threshold=0.5,
        confidence_metric='avg_trace_confidence'
    )
    print(f"Filtering results: {filtering_results}")

    # Save results
    env.make_dataset(results, push_to_hub=False)
    """

    print("\nDeepConf environment is ready!")
    print("\nTo run a full evaluation:")
    print("1. Set up your OpenAI-compatible client")
    print("2. Uncomment the evaluation code above")
    print("3. Run: python example_usage.py")

if __name__ == "__main__":
    main()
