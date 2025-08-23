import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


def calculate_token_confidence(logprobs: List[Dict[str, float]]) -> float:
    """
    Calculate token confidence as negative mean of alternative logprobs.
    Based on DeepConf paper: confidence = -mean(logprobs[1:])
    """
    if not logprobs or len(logprobs) < 2:
        return 0.0

    # logprobs[0] is the sampled token, use remaining as alternatives
    alt_logprobs = [item['logprob'] for item in logprobs[1:]]
    if not alt_logprobs:
        return 0.0

    return -np.mean(alt_logprobs)


def calculate_token_entropy(logprobs: List[Dict[str, float]]) -> float:
    """
    Calculate token entropy from logprobs.
    """
    if not logprobs:
        return 0.0

    probs = [math.exp(item['logprob']) for item in logprobs]
    probs = np.array(probs) / np.sum(probs)  # normalize

    # Calculate entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy


def calculate_group_confidence(confidences: List[float], group_size: int = 10) -> float:
    """
    Calculate group confidence by averaging confidence over groups of tokens.
    """
    if len(confidences) < group_size:
        return np.mean(confidences) if confidences else 0.0

    # Calculate group confidences
    group_confidences = []
    for i in range(0, len(confidences) - group_size + 1, group_size):
        group = confidences[i:i + group_size]
        group_confidences.append(np.mean(group))

    return np.mean(group_confidences) if group_confidences else 0.0


def calculate_bottom_10_percent_confidence(confidences: List[float]) -> float:
    """
    Calculate confidence of the bottom 10% of tokens.
    """
    if len(confidences) < 10:
        return np.mean(confidences) if confidences else 0.0

    # Sort confidences and take bottom 10%
    sorted_confidences = sorted(confidences)
    bottom_10_count = max(1, len(confidences) // 10)
    bottom_10 = sorted_confidences[:bottom_10_count]

    return np.mean(bottom_10)


def calculate_lowest_group_confidence(confidences: List[float], group_size: int = 10) -> float:
    """
    Calculate the confidence of the lowest-confidence group.
    """
    if len(confidences) < group_size:
        return np.mean(confidences) if confidences else 0.0

    # Calculate group confidences and return the minimum
    group_confidences = []
    for i in range(0, len(confidences) - group_size + 1, group_size):
        group = confidences[i:i + group_size]
        group_confidences.append(np.mean(group))

    return min(group_confidences) if group_confidences else 0.0


def calculate_tail_confidence(confidences: List[float], tail_percent: float = 0.1) -> float:
    """
    Calculate confidence of the tail (lowest) percentage of tokens.
    """
    if not confidences:
        return 0.0

    tail_count = max(1, int(len(confidences) * tail_percent))
    sorted_confidences = sorted(confidences)

    return np.mean(sorted_confidences[:tail_count])


def extract_confidence_features(completion: str, logprobs_data: Optional[List[List[Dict[str, float]]]] = None) -> Dict[str, float]:
    """
    Extract confidence features from completion and logprobs data.
    This simulates the confidence measurements described in DeepConf paper.
    """
    features = {}

    if logprobs_data and logprobs_data[0]:  # Check if we have logprobs
        # Extract token confidences
        token_confidences = []
        token_entropies = []

        for token_logprobs in logprobs_data[0]:
            if token_logprobs:
                confidence = calculate_token_confidence(token_logprobs)
                entropy = calculate_token_entropy(token_logprobs)
                token_confidences.append(confidence)
                token_entropies.append(entropy)

        if token_confidences:
            # Calculate various confidence metrics
            features['token_confidence'] = np.mean(token_confidences)
            features['group_confidence'] = calculate_group_confidence(token_confidences)
            features['bottom_10_percent_confidence'] = calculate_bottom_10_percent_confidence(token_confidences)
            features['lowest_group_confidence'] = calculate_lowest_group_confidence(token_confidences)
            features['tail_confidence'] = calculate_tail_confidence(token_confidences)
            features['avg_trace_confidence'] = np.mean(token_confidences)

            # Entropy-based metrics
            features['avg_entropy'] = np.mean(token_entropies) if token_entropies else 0.0
            features['max_entropy'] = max(token_entropies) if token_entropies else 0.0
        else:
            # Fallback values when no logprobs available
            features.update({
                'token_confidence': 0.0,
                'group_confidence': 0.0,
                'bottom_10_percent_confidence': 0.0,
                'lowest_group_confidence': 0.0,
                'tail_confidence': 0.0,
                'avg_trace_confidence': 0.0,
                'avg_entropy': 0.0,
                'max_entropy': 0.0
            })
    else:
        # Default values when no logprobs data
        features.update({
            'token_confidence': 0.0,
            'group_confidence': 0.0,
            'bottom_10_percent_confidence': 0.0,
            'lowest_group_confidence': 0.0,
            'tail_confidence': 0.0,
            'avg_trace_confidence': 0.0,
            'avg_entropy': 0.0,
            'max_entropy': 0.0
        })

    return features


def filter_traces_by_confidence(traces: List[Dict[str, Any]],
                                confidence_threshold: float,
                                confidence_metric: str = 'avg_trace_confidence') -> List[Dict[str, Any]]:
    """
    Filter traces based on confidence threshold (offline mode).
    """
    filtered_traces = []
    for trace in traces:
        confidence_features = extract_confidence_features(
            trace.get('completion', ''),
            trace.get('logprobs', None)
        )
        if confidence_features[confidence_metric] >= confidence_threshold:
            filtered_traces.append(trace)

    return filtered_traces


def confidence_weighted_majority_vote(traces: List[Dict[str, Any]],
                                     confidence_metric: str = 'avg_trace_confidence') -> str:
    """
    Perform confidence-weighted majority voting.
    """
    if not traces:
        return ""

    # Extract answers and their confidence weights
    answer_weights = defaultdict(float)
    answer_confidence_sums = defaultdict(float)

    for trace in traces:
        confidence_features = extract_confidence_features(
            trace.get('completion', ''),
            trace.get('logprobs', None)
        )

        # Parse answer from completion
        parsed_answer = extract_boxed_answer(trace.get('completion', ''))

        weight = confidence_features[confidence_metric]
        answer_weights[parsed_answer] += 1
        answer_confidence_sums[parsed_answer] += weight

    if not answer_weights:
        return ""

    # Return answer with highest confidence-weighted count
    best_answer = max(answer_weights.keys(),
                     key=lambda x: answer_weights[x] * answer_confidence_sums[x])
    return best_answer


def majority_vote(traces: List[Dict[str, Any]]) -> str:
    """
    Perform standard majority voting.
    """
    if not traces:
        return ""

    answer_counts = defaultdict(int)
    for trace in traces:
        parsed_answer = extract_boxed_answer(trace.get('completion', ''))
        answer_counts[parsed_answer] += 1

    if not answer_counts:
        return ""

    return max(answer_counts.keys(), key=lambda x: answer_counts[x])


def load_environment(
    dataset_name: str = "aime2025",
    confidence_metric: str = "avg_trace_confidence",
    confidence_threshold: float = 0.5,
    num_traces: int = 32,
    use_think: bool = True,
    system_prompt: str = BOXED_SYSTEM_PROMPT,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
):
    """
    Load DeepConf environment for confidence-aware reasoning evaluation.

    Args:
        dataset_name: Dataset to use (aime2024, aime2025, gpqa_diamond, etc.)
        confidence_metric: Which confidence metric to use for filtering
        confidence_threshold: Threshold for confidence filtering
        num_traces: Number of traces to generate per example
        use_think: Whether to use ThinkParser for reasoning traces
        system_prompt: System prompt to use
        num_train_examples: Limit training set size (-1 for all)
        num_eval_examples: Limit eval set size (-1 for all)
    """

    # Load dataset
    dataset = load_example_dataset(dataset_name, split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    eval_dataset = load_example_dataset(dataset_name, split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(min(num_eval_examples, len(eval_dataset))))

    # Setup parser
    if use_think:
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        parser = vf.Parser(extract_fn=extract_boxed_answer)

    def deepconf_correctness_reward_func(parser, completion, answer, **kwargs):
        """Reward function that evaluates correctness with confidence awareness."""
        response = parser.parse_answer(completion) or ""
        is_correct = 1.0 if response.strip() == answer.strip() else 0.0

        # Extract confidence features for additional metrics
        confidence_features = extract_confidence_features(completion)

        # Store confidence metrics in state for analysis
        state = kwargs.get('state', {})
        if 'confidence_metrics' not in state:
            state['confidence_metrics'] = []
        state['confidence_metrics'].append(confidence_features)

        return is_correct

    def confidence_quality_reward_func(parser, completion, answer, **kwargs):
        """Reward function based on confidence quality metrics."""
        confidence_features = extract_confidence_features(completion)

        # Higher confidence should generally correlate with better quality
        # Use a combination of confidence metrics as a quality score
        quality_score = (
            confidence_features.get('avg_trace_confidence', 0.0) * 0.5 +
            confidence_features.get('group_confidence', 0.0) * 0.3 +
            (1.0 - confidence_features.get('avg_entropy', 0.0)) * 0.2  # Lower entropy is better
        )

        return max(0.0, min(1.0, quality_score))

    def deepconf_filtering_reward_func(parser, completion, answer, **kwargs):
        """Reward function that applies DeepConf filtering logic."""
        response = parser.parse_answer(completion) or ""
        confidence_features = extract_confidence_features(completion)

        # Apply confidence threshold filtering
        confidence_score = confidence_features.get(confidence_metric, 0.0)
        passes_filter = 1.0 if confidence_score >= confidence_threshold else 0.0

        # Combine correctness and confidence filtering
        is_correct = 1.0 if response.strip() == answer.strip() else 0.0
        final_score = is_correct * passes_filter

        return final_score

    # Create rubric with multiple evaluation criteria
    rubric = vf.Rubric(
        funcs=[
            deepconf_correctness_reward_func,
            confidence_quality_reward_func,
            deepconf_filtering_reward_func,
            parser.get_format_reward_func()
        ],
        weights=[1.0, 0.2, 0.3, 0.1],  # Prioritize correctness, then filtering, then confidence quality
    )

    # Create environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return vf_env


# Additional utility functions for DeepConf-specific analysis
def analyze_confidence_correlation(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze correlation between confidence metrics and correctness.
    """
    if not results:
        return {}

    correct_confidences = []
    incorrect_confidences = []

    for result in results:
        confidence_metrics = result.get('state', {}).get('confidence_metrics', [])
        is_correct = result.get('reward', 0.0) > 0.5

        for metrics in confidence_metrics:
            confidence = metrics.get('avg_trace_confidence', 0.0)
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)

    analysis = {}
    if correct_confidences:
        analysis['correct_confidence_mean'] = np.mean(correct_confidences)
        analysis['correct_confidence_std'] = np.std(correct_confidences)

    if incorrect_confidences:
        analysis['incorrect_confidence_mean'] = np.mean(incorrect_confidences)
        analysis['incorrect_confidence_std'] = np.std(incorrect_confidences)

    # Calculate separation between correct and incorrect
    if correct_confidences and incorrect_confidences:
        analysis['confidence_separation'] = (
            analysis['correct_confidence_mean'] - analysis['incorrect_confidence_mean']
        )

    return analysis


def simulate_deepconf_filtering(results: List[Dict[str, Any]],
                               confidence_threshold: float = 0.5,
                               confidence_metric: str = 'avg_trace_confidence') -> Dict[str, float]:
    """
    Simulate DeepConf filtering on evaluation results.
    """
    if not results:
        return {}

    # Simulate filtering
    filtered_results = []
    for result in results:
        confidence_metrics = result.get('state', {}).get('confidence_metrics', [])
        if not confidence_metrics:
            continue

        # Check if result passes confidence filter
        max_confidence = max(m.get(confidence_metric, 0.0) for m in confidence_metrics)
        if max_confidence >= confidence_threshold:
            filtered_results.append(result)

    # Calculate filtering statistics
    original_count = len(results)
    filtered_count = len(filtered_results)

    if original_count == 0:
        return {}

    # Calculate accuracy before and after filtering
    original_accuracy = sum(1 for r in results if r.get('reward', 0.0) > 0.5) / original_count
    filtered_accuracy = sum(1 for r in filtered_results if r.get('reward', 0.0) > 0.5) / max(filtered_count, 1)

    return {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'filtering_ratio': filtered_count / original_count,
        'original_accuracy': original_accuracy,
        'filtered_accuracy': filtered_accuracy,
        'accuracy_improvement': filtered_accuracy - original_accuracy
    }
