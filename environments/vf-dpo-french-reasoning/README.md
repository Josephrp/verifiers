# DPO French Reasoning Environment

A Verifiers environment for DPO (Direct Preference Optimization) training with French reasoning traces.

## Features

- **Multi-Dataset Loading**: Combines multiple datasets into unified format
- **French Thinking Generation**: Generates French reasoning traces when missing
- **Automatic Translation**: Translates non-French thinking to French
- **4-Column DPO Format**: prompt | thinking | accepted | rejected
- **Preference-Based Rewards**: Evaluates responses against preferred outcomes
- **Multi-Turn Interactions**: Supports complex reasoning workflows

## Data Format

The environment processes datasets into this format:

- **prompt**: The input question or task
- **thinking**: French reasoning trace (generated if missing, translated if not in French)
- **accepted**: Preferred/correct response
- **rejected**: Non-preferred/incorrect response

## Usage

### Basic Usage

```python
import verifiers as vf

# Load environment with default datasets
env = vf.load_environment(env_id="vf-dpo-french-reasoning")

# Load with custom datasets
env = vf.load_environment(
    env_id="vf-dpo-french-reasoning",
    dataset_names=["gsm8k", "math_dataset", "your_custom_dataset"]
)
```

### Custom Configuration

```python
env = vf.load_environment(
    env_id="vf-dpo-french-reasoning",
    local_model="microsoft/DialoGPT-medium",  # Model for thinking generation
    max_turns=5,                             # Maximum interaction turns
    num_train_examples=1000,                 # Training examples to use
    num_eval_examples=200                    # Evaluation examples
)
```

## Installation

Install the environment:

```bash
vf-install vf-dpo-french-reasoning
```

Or install from source:

```bash
vf-install vf-dpo-french-reasoning --from-repo
```

## Dependencies

- `verifiers`: Core framework
- `transformers`: For local model loading
- `torch`: Deep learning framework
- `datasets`: Data loading
- `langdetect`: Language detection
- `deep-translator`: Translation service

## Training Integration

### With Verifiers GRPO Trainer

```python
import verifiers as vf

# Load environment
env = vf.load_environment(env_id="vf-dpo-french-reasoning")

# Load model and tokenizer
model, tokenizer = vf.get_model_and_tokenizer("your-model-name")

# Configure training
training_args = vf.grpo_defaults(
    run_name="dpo-french-reasoning",
    learning_rate=1e-5,
    num_generations=8,
    per_device_train_batch_size=4
)

# Create trainer
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=training_args
)

# Train
trainer.train()
```

### Direct DPO Training

```python
from trl import DPOTrainer
from transformers import TrainingArguments

# Prepare dataset in DPO format
train_dataset = env.dataset

# Configure DPO training
training_args = TrainingArguments(
    output_dir="./dpo-french-reasoning",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    beta=0.1,  # DPO temperature parameter
)

# Train
dpo_trainer.train()
```

## Data Processing Pipeline

1. **Dataset Loading**: Load multiple datasets from Hugging Face
2. **Thinking Generation**: Use local model to generate reasoning when missing
3. **Language Detection**: Check if thinking is in French
4. **Translation**: Translate non-French thinking to French
5. **Preference Pairs**: Generate accepted/rejected response pairs
6. **Format Conversion**: Convert to 4-column DPO format

## Environment Parameters

- `dataset_names`: List of dataset names to load
- `local_model`: Hugging Face model for thinking generation
- `max_turns`: Maximum turns per interaction
- `num_train_examples`: Number of training examples (-1 for all)
- `num_eval_examples`: Number of evaluation examples (-1 for all)

## Evaluation

```python
# Evaluate with API model
results = env.evaluate(
    client=OpenAI(),
    model="gpt-4",
    num_examples=100,
    rollouts_per_example=3
)

# Save results
env.make_dataset(results, push_to_hub=True, hub_name="your-eval-results")
```

## Customization

### Custom Thinking Generator

```python
from environments.vf_dpo_french_reasoning import FrenchThinkingGenerator

class CustomThinkingGenerator(FrenchThinkingGenerator):
    def generate_thinking(self, prompt: str) -> str:
        # Your custom thinking generation logic
        return super().generate_thinking(prompt)

# Use custom generator
env = load_environment(thinking_generator=CustomThinkingGenerator())
```

### Custom Reward Functions

```python
def custom_preference_reward(prompt, completion, answer, state, **kwargs):
    # Your custom preference logic
    return score

env.rubric = vf.Rubric(
    funcs=[custom_preference_reward, env._french_quality_reward],
    weights=[1.0, 0.5]
)
```

## Examples

See the `examples/` directory for complete training scripts and evaluation workflows.
