#!/usr/bin/env python3
"""
GRPO Training with DPO-style Preferences for French Reasoning

This script combines GRPO (Generative Proximal Policy Optimization) with
DPO-inspired preference learning for French reasoning tasks.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num-processes 2 \
        --config-file configs/zero3.yaml examples/grpo/train_grpo_dpo_french.py

Requirements:
    pip install verifiers[all] flash-attn
"""

import verifiers as vf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import numpy as np

# Environment configuration
env = vf.load_environment(
    env_id="vf-dpo-french-reasoning",
    dataset_names=["gsm8k", "math_dataset"],  # Multiple datasets
    local_model="microsoft/DialoGPT-medium",
    max_turns=3,
    num_train_examples=1000
)

# Model configuration
model_name = "Qwen/Qwen2.5-7B-Instruct"
run_name = f"grpo-dpo-french-{model_name.split('/')[-1].lower()}"

# Load model and tokenizer
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Enhanced GRPO configuration with DPO-style preferences
training_args = vf.grpo_defaults(run_name=run_name)

# Training hyperparameters optimized for French reasoning
training_args.per_device_train_batch_size = 4
training_args.num_generations = 8  # Generate multiple responses for preference learning
training_args.gradient_accumulation_steps = 4
training_args.max_tokens = 1024
training_args.max_seq_len = 1024
training_args.learning_rate = 1e-6
training_args.lr_scheduler_type = "cosine"
training_args.warmup_steps = 100
training_args.max_steps = 1000
training_args.eval_strategy = "steps"
training_args.eval_steps = 100
training_args.save_strategy = "steps"
training_args.save_steps = 200
training_args.logging_steps = 10

# LoRA configuration for efficient training
lora_config = vf.lora_defaults(
    r=32,  # Higher rank for complex reasoning
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05
)


class DPOEnhancedGRPOTrainer(vf.GRPOTrainer):
    """Enhanced GRPO trainer with DPO-style preference learning"""

    def __init__(self, preference_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.preference_weight = preference_weight

    def compute_preference_loss(self, generations, rewards):
        """Compute DPO-style preference loss from multiple generations"""
        if len(generations) < 2:
            return 0.0

        # Simple preference learning: higher reward generations get positive preference
        preference_loss = 0.0
        for i in range(len(generations) - 1):
            for j in range(i + 1, len(generations)):
                if rewards[i] > rewards[j]:
                    # Generation i is preferred over generation j
                    preference_loss += torch.log(1 / (1 + torch.exp(rewards[j] - rewards[i])))
                elif rewards[j] > rewards[i]:
                    # Generation j is preferred over generation i
                    preference_loss += torch.log(1 / (1 + torch.exp(rewards[i] - rewards[j])))

        return self.preference_weight * preference_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override loss computation to include preference learning"""
        # Original GRPO loss
        outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # Add DPO-style preference loss
        if hasattr(outputs, 'logits') and inputs.get('generations') is not None:
            preference_loss = self.compute_preference_loss(
                inputs['generations'],
                inputs.get('rewards', [])
            )
            outputs['loss'] = outputs['loss'] + preference_loss

        return outputs


# Create enhanced trainer
trainer = DPOEnhancedGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=training_args,
    peft_config=lora_config,
    preference_weight=0.3  # Weight for DPO-style preference learning
)

# Custom reward function that incorporates French reasoning preferences
def enhanced_french_reward_func(parser, completion, answer, state, **kwargs):
    """Enhanced reward function for French reasoning with preference learning"""
    reward = 0.0

    # Base correctness reward
    response = parser.parse_answer(completion) or ""
    if response == answer:
        reward += 2.0

    # French language quality reward
    french_indicators = ['raisonnons', 'réfléchissons', 'étape', 'premièrement', 'ensuite']
    for message in completion:
        if message['role'] == 'assistant':
            content = message['content'].lower()
            for indicator in french_indicators:
                if indicator in content:
                    reward += 0.5

    # Preference-based reward (if available)
    if 'accepted' in state and 'rejected' in state:
        accepted_keywords = set(state['accepted'].lower().split())
        rejected_keywords = set(state['rejected'].lower().split())
        response_keywords = set(response.lower().split())

        # Reward for matching accepted keywords, penalize rejected keywords
        accepted_matches = len(accepted_keywords.intersection(response_keywords))
        rejected_matches = len(rejected_keywords.intersection(response_keywords))

        reward += 0.3 * accepted_matches
        reward -= 0.2 * rejected_matches

    return reward

# Update environment rubric with enhanced reward
env.rubric = vf.Rubric(
    funcs=[enhanced_french_reward_func, env.rubric.funcs[1]],  # Keep format reward
    weights=[1.0, 0.1]
)

# Training callback for monitoring preference learning
class PreferenceLearningCallback:
    def __init__(self):
        self.preference_scores = []

    def on_step_end(self, args, state, control, model, tokenizer, eval_dataloader=None, **kwargs):
        if state.global_step % 50 == 0:
            # Evaluate preference learning progress
            print(f"Step {state.global_step}: Evaluating preference learning...")

            # Simple evaluation of French reasoning quality
            eval_results = env.evaluate(
                model=model,
                tokenizer=tokenizer,
                num_examples=10,
                max_new_tokens=256
            )

            avg_reward = np.mean([r.get('reward', 0) for r in eval_results.data])
            self.preference_scores.append((state.global_step, avg_reward))
            print(".3f")

# Add callback
preference_callback = PreferenceLearningCallback()
trainer.add_callback(preference_callback)

# Start training
print("Starting GRPO training with DPO-style preference learning...")
print(f"Training on {len(env.dataset)} examples")
print(f"Model: {model_name}")
print(f"Environment: {len(env.dataset)} training examples")

trainer.train()

# Save final model
trainer.save_model()
print("Training completed! Model saved.")

# Final evaluation
print("Running final evaluation...")
final_results = env.evaluate(
    model=model,
    tokenizer=tokenizer,
    num_examples=100,
    max_new_tokens=256
)

print("=== FINAL EVALUATION RESULTS ===")
print(f"Average Reward: {np.mean([r.get('reward', 0) for r in final_results.data]):.3f}")
print(f"Total Examples: {len(final_results.data)}")

# Show some examples
print("\\n=== SAMPLE RESPONSES ===")
for i in range(min(3, len(final_results.data))):
    example = final_results.data[i]
    print(f"\\nExample {i+1}:")
    print(f"Prompt: {example['prompt']}")
    print(f"Response: {example['completion']}")
    print(f"Reward: {example.get('reward', 0):.3f}")
