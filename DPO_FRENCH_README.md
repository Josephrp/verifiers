# DPO French Reasoning Environment - Complete Guide

This guide explains how to set up and use the DPO (Direct Preference Optimization) French reasoning environment I created for you.

## Overview

The system implements a complete pipeline for training LLMs with French reasoning capabilities using preference learning:

1. **Multi-Dataset Loading**: Combines multiple datasets into unified format
2. **Thinking Generation**: Generates French reasoning traces when missing
3. **Language Translation**: Translates non-French content to French
4. **4-Column DPO Format**: Converts data to `prompt|thinking|accepted|rejected`
5. **Preference Training**: Uses DPO and GRPO for preference-based learning
6. **Comprehensive Evaluation**: Validates all components of the pipeline

## Quick Start

### 1. One-Command Setup

```bash
# Run the complete pipeline
python setup_dpo_french_environment.py --all
```

This will:
- Install the environment
- Set up training data
- Test the environment
- Run DPO training
- Evaluate the results

### 2. Step-by-Step Setup

```bash
# 1. Install environment
python setup_dpo_french_environment.py --install

# 2. Test environment
python setup_dpo_french_environment.py --quick-test

# 3. Run DPO training
python setup_dpo_french_environment.py --train-dpo

# 4. Evaluate results
python setup_dpo_french_environment.py --evaluate
```

## Architecture

### Core Components

```
📁 environments/vf-dpo-french-reasoning/
├── vf_dpo_french_reasoning.py      # Main environment implementation
├── pyproject.toml                  # Dependencies and metadata
└── README.md                       # Environment documentation

📁 examples/dpo/
├── train_dpo_french_reasoning.py   # DPO training script
└── evaluate_dpo_french.py          # Evaluation script

📁 examples/grpo/
└── train_grpo_dpo_french.py        # GRPO + DPO training script

📄 setup_dpo_french_environment.py   # Main setup script
```

### Data Flow

1. **Dataset Loading** → Multiple datasets (GSM8K, MATH, etc.)
2. **Thinking Processing** → Generate/translate French reasoning
3. **Format Conversion** → 4-column DPO format
4. **Model Training** → DPO or GRPO with preference learning
5. **Evaluation** → Comprehensive metrics and validation

## Environment Details

### FrenchThinkingGenerator Class

```python
# Key features:
- Local model-based thinking generation
- French language detection
- Automatic translation to French
- Configurable generation parameters
```

### DPOFrenchReasoningEnvironment Class

```python
# Core functionality:
- Multi-turn interaction protocol
- French quality reward functions
- Preference-based evaluation
- State management for reasoning traces
```

### Data Format

The environment processes data into this format:

```python
{
    'prompt': 'Question text',
    'thinking': 'French reasoning trace',
    'accepted': 'Preferred answer',
    'rejected': 'Non-preferred answer',
    'original_answer': 'Ground truth'
}
```

## Training Options

### 1. DPO Training (Recommended)

```bash
# Standard DPO training
python examples/dpo/train_dpo_french_reasoning.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset_names gsm8k \
    --num_train_examples 1000 \
    --learning_rate 5e-7 \
    --use_peft
```

**Pros:**
- Direct preference optimization
- Faster training
- Good for preference learning
- Lower computational requirements

### 2. GRPO Training with DPO Preferences

```bash
# GRPO with enhanced preference learning
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml \
    examples/grpo/train_grpo_dpo_french.py
```

**Pros:**
- Combines RL with preference learning
- Better exploration
- More stable training
- Enhanced reasoning capabilities

### 3. Multi-GPU Training

```bash
# For larger models with multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num-processes 4 \
    --config-file configs/zero3.yaml \
    examples/grpo/train_grpo_dpo_french.py
```

## Evaluation and Validation

### Comprehensive Evaluation

```bash
python examples/dpo/evaluate_dpo_french.py \
    --model ./dpo-french-reasoning-output \
    --baseline_model microsoft/DialoGPT-medium \
    --num_examples 100 \
    --verbose \
    --output_file evaluation_results.json
```

### Metrics Evaluated

- **French Language Quality**: Language detection and structure
- **Preference Alignment**: Alignment with accepted/rejected pairs
- **Reasoning Completeness**: Thinking trace quality and completeness
- **Answer Correctness**: Mathematical accuracy
- **Response Quality**: Overall response structure

### Sample Evaluation Output

```
=== DETAILED RESULTS ===

Trained Model Results:
  Average French Score: 0.950
  Average Structure Score: 0.780
  Average Preference Score: 0.650
  Average Correctness: 0.720
  Average Completeness: 0.680

=== MODEL COMPARISON ===

Trained Model:
  avg_french_scores_improvement: +0.850
  avg_preference_scores_improvement: +0.400
  avg_correctness_scores_improvement: +0.350
```

## Customization

### Custom Datasets

```python
# Add your own datasets
env = vf.load_environment(
    env_id="vf-dpo-french-reasoning",
    dataset_names=["your-dataset-1", "your-dataset-2"],
    local_model="your-model"
)
```

### Custom Thinking Generator

```python
from environments.vf_dpo_french_reasoning import FrenchThinkingGenerator

class CustomThinkingGenerator(FrenchThinkingGenerator):
    def generate_thinking(self, prompt: str) -> str:
        # Your custom logic
        return super().generate_thinking(prompt)

env = load_environment(thinking_generator=CustomThinkingGenerator())
```

### Custom Reward Functions

```python
def custom_french_reward(prompt, completion, answer, state, **kwargs):
    # Your custom French evaluation logic
    return score

env.rubric = vf.Rubric(
    funcs=[custom_french_reward, env._french_quality_reward],
    weights=[1.0, 0.5]
)
```

## Advanced Features

### Multi-Turn Interactions

The environment supports complex multi-turn reasoning:

```python
env = DPOFrenchReasoningEnvironment(
    dataset=dataset,
    max_turns=5,  # Allow up to 5 turns
    thinking_generator=thinking_generator
)
```

### Custom Translation Services

```python
# Use different translation service
env.thinking_generator.translator = YourTranslator()
```

### Enhanced Evaluation

```python
# Detailed evaluation with custom metrics
results = env.evaluate(
    model=model,
    tokenizer=tokenizer,
    num_examples=100,
    max_new_tokens=256,
    custom_metrics=['french_quality', 'preference_alignment']
)
```

## Troubleshooting

### Common Issues

1. **Environment Installation Failed**
   ```bash
   # Manual installation
   cd environments/vf-dpo-french-reasoning
   pip install -e .
   ```

2. **Translation Issues**
   ```python
   # Check translation service
   from deep_translator import GoogleTranslator
   translator = GoogleTranslator(source='auto', target='fr')
   print(translator.translate("Hello world"))
   ```

3. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --per_device_train_batch_size 1
   --gradient_accumulation_steps 8
   ```

4. **Dataset Loading Errors**
   ```python
   # Use local datasets
   dataset = load_dataset('path/to/local/dataset')
   ```

### Performance Tips

1. **Use PEFT for Memory Efficiency**
   ```python
   --use_peft --lora_r 16 --lora_alpha 32
   ```

2. **Optimize Translation**
   ```python
   # Cache translations to avoid repeated API calls
   env.thinking_generator.enable_cache = True
   ```

3. **Batch Processing**
   ```python
   --per_device_train_batch_size 4
   --gradient_accumulation_steps 4
   ```

## Integration with Verifiers

The environment integrates seamlessly with the verifiers framework:

```python
# Use with existing verifiers workflows
env = vf.load_environment(env_id="vf-dpo-french-reasoning")

# Standard evaluation
results = env.evaluate(client=openai_client, model="gpt-4")

# Custom rollouts
async for rollout in env.async_rollout(client, model, prompt):
    process_rollout(rollout)
```

## Future Enhancements

Potential improvements and extensions:

1. **Multi-Language Support**: Extend beyond French
2. **Advanced Translation**: Use specialized translation models
3. **Preference Model**: Train dedicated preference classifier
4. **Interactive Training**: Human-in-the-loop preference collection
5. **Knowledge Integration**: Add fact-checking and verification

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run the quick test: `python setup_dpo_french_environment.py --quick-test`
3. Check the logs and error messages
4. Verify dependencies: `pip list | grep -E "(verifiers|transformers|torch)"`

## Citation

If you use this implementation in your research:

```
@software{dpo_french_reasoning_2025,
  title={DPO French Reasoning Environment},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/verifiers-dpo-french}
}
```

---

**Note**: This implementation provides a complete, production-ready system for DPO training with French reasoning. The modular design allows easy customization and extension for your specific use cases.
