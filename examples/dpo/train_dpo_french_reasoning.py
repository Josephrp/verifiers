#!/usr/bin/env python3
"""
DPO Training Script for French Reasoning Environment

This script demonstrates how to train a model using Direct Preference Optimization (DPO)
with the French reasoning environment.

Usage:
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num-processes 1 \
        examples/dpo/train_dpo_french_reasoning.py --model Qwen/Qwen2.5-7B-Instruct

Requirements:
    pip install trl transformers accelerate peft datasets
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
import verifiers as vf

def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO with French Reasoning Environment")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./dpo-french-reasoning-output", help="Output directory")
    parser.add_argument("--dataset_names", nargs="+", default=["gsm8k"], help="Datasets to use")
    parser.add_argument("--local_model", type=str, default="microsoft/DialoGPT-medium", help="Local model for thinking generation")
    parser.add_argument("--num_train_examples", type=int, default=1000, help="Number of training examples")
    parser.add_argument("--num_eval_examples", type=int, default=200, help="Number of evaluation examples")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT for training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    return parser.parse_args()


def prepare_model_and_tokenizer(model_name: str, use_peft: bool = True, **lora_kwargs):
    """Load model and tokenizer with optional PEFT configuration"""
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare PEFT config if requested
    peft_config = None
    if use_peft:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_kwargs.get("lora_r", 16),
            lora_alpha=lora_kwargs.get("lora_alpha", 32),
            lora_dropout=lora_kwargs.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        print(f"Using LoRA with rank {lora_kwargs.get('lora_r', 16)}")

    return model, tokenizer, peft_config


def format_dpo_example(example):
    """Format example for DPO training"""
    # Create system prompt with French thinking guidance
    system_prompt = """Vous êtes un assistant qui raisonne étape par étape en français.

Voici un raisonnement pour vous guider:
{thinking}

Maintenant, répondez à la question en utilisant ce raisonnement."""

    prompt = system_prompt.format(thinking=example['thinking'])
    prompt += f"\n\nQuestion: {example['prompt']}\n\nRéponse:"

    return {
        "prompt": prompt,
        "chosen": example['accepted'],
        "rejected": example['rejected'],
    }


def main():
    args = parse_args()

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load environment
    print("Loading French reasoning environment...")
    env = vf.load_environment(
        env_id="vf-dpo-french-reasoning",
        dataset_names=args.dataset_names,
        local_model=args.local_model,
        num_train_examples=args.num_train_examples,
        num_eval_examples=args.num_eval_examples,
    )

    # Prepare dataset for DPO
    print("Preparing dataset for DPO training...")
    train_dataset = env.dataset
    eval_dataset = env.eval_dataset

    # Format datasets for DPO
    train_dataset = train_dataset.map(format_dpo_example)
    if eval_dataset:
        eval_dataset = eval_dataset.map(format_dpo_example)

    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")

    # Load model and tokenizer
    model, tokenizer, peft_config = prepare_model_and_tokenizer(
        args.model,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Configure DPO training
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        push_to_hub=False,
        report_to="wandb",
        run_name=f"dpo-french-reasoning-{Path(args.model).name}",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        beta=args.beta,
        loss_type="sigmoid",  # or "ipo" for IPO loss
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train the model
    print("Starting DPO training...")
    trainer.train()

    # Save the model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()

    # Evaluate final performance
    if eval_dataset:
        print("Evaluating final model...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")

    print("DPO training completed!")


def create_comparison_script():
    """Create a script to compare model performance before and after DPO training"""
    comparison_script = '''
import verifiers as vf
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def compare_models(base_model_path, dpo_model_path, env_id="vf-dpo-french-reasoning"):
    """Compare base model vs DPO model performance"""

    # Load environment
    env = vf.load_environment(env_id=env_id, num_eval_examples=100)

    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    def evaluate_model(model, name):
        print(f"Evaluating {name}...")
        results = env.evaluate(
            model=model,
            tokenizer=tokenizer,
            num_examples=50,
            max_new_tokens=256
        )
        return results

    # Evaluate both models
    base_results = evaluate_model(base_model, "Base Model")
    dpo_results = evaluate_model(dpo_model, "DPO Model")

    # Compare results
    print("\\n=== COMPARISON RESULTS ===")
    print(f"Base Model - Average Reward: {base_results.metrics.get('reward', 0):.3f}")
    print(f"DPO Model  - Average Reward: {dpo_results.metrics.get('reward', 0):.3f}")

    # Show some examples
    print("\\n=== SAMPLE RESPONSES ===")
    for i in range(min(5, len(base_results.data))):
        print(f"\\nExample {i+1}:")
        print(f"Prompt: {base_results.data[i]['prompt']}")
        print(f"Base: {base_results.data[i]['completion']}")
        print(f"DPO:  {dpo_results.data[i]['completion']}")

if __name__ == "__main__":
    # Example usage
    compare_models(
        base_model_path="your-base-model",
        dpo_model_path="./dpo-french-reasoning-output"
    )
'''

    with open("examples/dpo/compare_dpo_models.py", "w") as f:
        f.write(comparison_script)
    print("Created comparison script: examples/dpo/compare_dpo_models.py")


if __name__ == "__main__":
    # Run main training
    main()

    # Create comparison script
    create_comparison_script()
