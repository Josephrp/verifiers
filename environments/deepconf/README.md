# deepconf

### Overview
- **Environment ID**: `deepconf`
- **Short description**: DeepConf environment for confidence-aware LLM reasoning evaluation based on the DeepConf paper [https://arxiv.org/html/2508.15260v1](https://arxiv.org/html/2508.15260v1)
- **Tags**: deepconf, confidence, reasoning, single-turn, math, eval

### DeepConf Paper Implementation
This environment implements the confidence-based reasoning trace filtering methods described in the DeepConf paper. It provides:

- **Confidence Measurement**: Multiple confidence metrics including token confidence, group confidence, bottom-10% confidence, and tail confidence
- **Offline Filtering**: Confidence-based filtering of reasoning traces after generation
- **Online Thinking**: Real-time confidence monitoring during generation (when supported by inference server)
- **Majority Voting**: Standard and confidence-weighted majority voting for answer aggregation

### Datasets
- **Primary datasets**: Supports all mathematical reasoning datasets from the DeepConf paper:
  - `aime2024` - AIME 2024 competition problems
  - `aime2025` - AIME 2025 competition problems
  - `gpqa_diamond` - GPQA Diamond dataset
  - `amc2023` - AMC 2023 problems
  - Plus all other datasets supported by `verifiers.utils.data_utils.load_example_dataset()`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to full train/test

### Task
- **Type**: single-turn with confidence analysis
- **Parser**: `ThinkParser` with boxed answer extraction
- **Rubric overview**: Multi-criteria evaluation combining correctness, confidence quality, and format adherence

### Confidence Metrics Implemented

1. **Token Confidence**: Negative mean of alternative token logprobs
2. **Group Confidence**: Average confidence across token groups
3. **Bottom 10% Confidence**: Confidence of the lowest 10% of tokens
4. **Lowest Group Confidence**: Minimum confidence among token groups
5. **Tail Confidence**: Confidence of the tail percentage of tokens
6. **Average Trace Confidence**: Overall confidence of the reasoning trace
7. **Entropy-based metrics**: Token entropy and max entropy measurements

### Quickstart

Run an evaluation with default DeepConf settings:

```bash
# Install the environment
vf-install deepconf

# Quick evaluation with AIME 2025
uv run vf-eval deepconf -m gpt-4.1-mini --env-args '{"dataset_name": "aime2025"}'
```

Configure model and confidence parameters:

```bash
uv run vf-eval deepconf \
  -m gpt-4.1-mini \
  -n 10 -r 3 -t 2048 -T 0.7 \
  -a '{
    "dataset_name": "aime2025",
    "confidence_metric": "avg_trace_confidence",
    "confidence_threshold": 0.5,
    "num_traces": 32
  }'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"aime2025"` | Dataset to use (aime2024, aime2025, gpqa_diamond, etc.) |
| `confidence_metric` | str | `"avg_trace_confidence"` | Confidence metric for filtering |
| `confidence_threshold` | float | `0.5` | Threshold for confidence filtering |
| `num_traces` | int | `32` | Number of traces to generate per example |
| `use_think` | bool | `true` | Whether to use ThinkParser for reasoning traces |
| `num_train_examples` | int | `-1` | Limit training set size (-1 for all) |
| `num_eval_examples` | int | `-1` | Limit eval set size (-1 for all) |

### Confidence Metrics Available

| Metric | Description |
| ------ | ----------- |
| `token_confidence` | Average confidence across all tokens |
| `group_confidence` | Group-based confidence measurement |
| `bottom_10_percent_confidence` | Confidence of bottom 10% of tokens |
| `lowest_group_confidence` | Minimum group confidence |
| `tail_confidence` | Tail percentage confidence |
| `avg_trace_confidence` | Overall trace confidence |
| `avg_entropy` | Average token entropy |
| `max_entropy` | Maximum token entropy |

### DeepConf Methods

#### Offline Filtering
```python
from environments.deepconf.deepconf import filter_traces_by_confidence

# Filter traces below confidence threshold
high_confidence_traces = filter_traces_by_confidence(
    traces=generated_traces,
    confidence_threshold=0.5,
    confidence_metric='avg_trace_confidence'
)
```

#### Confidence-Weighted Majority Voting
```python
from environments.deepconf.deepconf import confidence_weighted_majority_vote

# Get final answer using confidence-weighted voting
final_answer = confidence_weighted_majority_vote(
    traces=traces,
    confidence_metric='avg_trace_confidence'
)
```

#### Confidence Analysis
```python
from environments.deepconf.deepconf import analyze_confidence_correlation

# Analyze correlation between confidence and correctness
analysis = analyze_confidence_correlation(evaluation_results)
print(f"Confidence separation: {analysis.get('confidence_separation', 0.0):.3f}")
```

### Advanced Usage

#### Custom Confidence Metrics
```python
# You can implement custom confidence metrics by extending the environment
def custom_confidence_metric(completion, logprobs_data):
    # Your custom logic here
    return custom_score

# Then use it in the environment arguments
env_args = {
    "confidence_metric": "custom_metric",
    # ... other args
}
```

#### Integration with vLLM for Online DeepConf
For online confidence monitoring during generation, you need vLLM with DeepConf modifications:

```python
# Enable online confidence monitoring (requires modified vLLM)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="your-model",
    messages=messages,
    extra_body={
        "vllm_xargs": {
            "enable_conf": True,
            "window_size": 2048,
            "threshold": 0.5
        }
    }
)
```

### Paper Reproduction

This environment is designed to reproduce the key experiments from the DeepConf paper:

- **Figure 1**: Parallel thinking vs DeepConf filtering
- **Table 2**: Performance improvements on AIME 2025
- **Figure 3**: Token reduction vs accuracy trade-offs
- **Table 4**: Cross-model performance improvements

### Performance Expectations

Based on the paper, you can expect:
- **AIME 2025**: Up to 99.9% accuracy with DeepConf@512 vs 97.0% with majority voting
- **Token Reduction**: Up to 84.7% reduction in generated tokens
- **GPQA-Diamond**: Significant improvements on challenging reasoning tasks

### Implementation Notes

- **Logprobs Requirement**: For full confidence analysis, your inference server must provide `logprobs` and `top_logprobs` in responses
- **vLLM Integration**: Online DeepConf requires the vLLM modifications described in Appendix G of the paper
- **Memory Usage**: Confidence analysis requires storing additional metadata for each generated token

### Troubleshooting

#### No Confidence Data
If you see all confidence metrics as 0.0:
- Ensure your model/inference server supports `logprobs=True`
- Check that `top_logprobs` is set to a value >= 2
- Verify that the response includes logprobs data

#### Poor Filtering Performance
- Try different confidence metrics (e.g., `group_confidence` instead of `avg_trace_confidence`)
- Adjust the confidence threshold based on your model's output patterns
- Consider using `confidence_weighted_majority_vote` instead of threshold filtering

#### Memory Issues with Large Traces
- Reduce `num_traces` parameter
- Use smaller `window_size` for confidence calculations
- Consider offline filtering instead of online monitoring

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval deepconf -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
