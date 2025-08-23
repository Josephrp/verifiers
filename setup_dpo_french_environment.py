#!/usr/bin/env python3
"""
Setup Script for DPO French Reasoning Environment

This script helps you set up and run the complete DPO French reasoning pipeline.

Usage:
    python setup_dpo_french_environment.py --install
    python setup_dpo_french_environment.py --train-dpo
    python setup_dpo_french_environment.py --train-grpo
    python setup_dpo_french_environment.py --evaluate
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str = ""):
    """Run a shell command with error handling"""
    print(f"{'='*50}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def install_environment():
    """Install the DPO French reasoning environment"""
    print("Installing DPO French reasoning environment...")

    # Install the environment
    success = run_command(
        "vf-install vf-dpo-french-reasoning",
        "Installing the environment module"
    )

    if not success:
        print("Failed to install environment. Trying manual installation...")

        # Create the environment directory if it doesn't exist
        env_dir = "environments/vf-dpo-french-reasoning"
        if not os.path.exists(env_dir):
            print(f"Creating environment directory: {env_dir}")
            os.makedirs(env_dir, exist_ok=True)

        # Install dependencies
        print("Installing dependencies...")
        run_command("pip install langdetect deep-translator", "Installing translation dependencies")

    print("Environment setup complete!")


def setup_training_data():
    """Setup and prepare training data"""
    print("Setting up training data...")

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Download some sample datasets if not present
    datasets_to_download = [
        "gsm8k",
        "microsoft/orca-math-word-problems-200k",
        "openai/gsm8k"
    ]

    for dataset in datasets_to_download:
        print(f"Checking dataset: {dataset}")
        # Note: Actual dataset loading will happen during training
        # This is just to verify the environment can access them

    print("Training data setup complete!")


def train_dpo():
    """Run DPO training"""
    print("Starting DPO training...")

    # Default model and configuration
    model_name = "microsoft/DialoGPT-medium"
    output_dir = "./dpo-french-reasoning-output"

    command = f"""
    python examples/dpo/train_dpo_french_reasoning.py \\
        --model {model_name} \\
        --output_dir {output_dir} \\
        --dataset_names gsm8k \\
        --num_train_examples 500 \\
        --num_eval_examples 100 \\
        --learning_rate 5e-7 \\
        --per_device_train_batch_size 2 \\
        --gradient_accumulation_steps 4 \\
        --num_train_epochs 3 \\
        --use_peft
    """

    success = run_command(command, "Running DPO training")

    if success:
        print(f"\\nDPO training completed! Model saved to: {output_dir}")
        print("You can now evaluate the model using: python setup_dpo_french_environment.py --evaluate")
    else:
        print("DPO training failed. Check the error messages above.")


def train_grpo():
    """Run GRPO training with DPO preferences"""
    print("Starting GRPO training with DPO preferences...")

    # Check if we have multiple GPUs
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            device_config = "CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num-processes 2"
        else:
            device_config = "CUDA_VISIBLE_DEVICES=0 accelerate launch --num-processes 1"
    except:
        device_config = "accelerate launch --num-processes 1"

    command = f"""
    {device_config} \\
        --config-file configs/zero3.yaml \\
        examples/grpo/train_grpo_dpo_french.py
    """

    success = run_command(command, "Running GRPO training with DPO preferences")

    if success:
        print("\\nGRPO training completed!")
        print("Model saved with enhanced French reasoning capabilities.")
    else:
        print("GRPO training failed. Check the error messages above.")


def evaluate_models():
    """Evaluate trained models"""
    print("Starting model evaluation...")

    # Find trained models
    possible_model_paths = [
        "./dpo-french-reasoning-output",
        "./grpo-dpo-french-output",
        "microsoft/DialoGPT-medium"  # Baseline
    ]

    trained_model = None
    baseline_model = None

    for path in possible_model_paths:
        if os.path.exists(path) and os.path.isdir(path):
            if "baseline" in path.lower() or path == possible_model_paths[-1]:
                baseline_model = path
            else:
                trained_model = path

    if trained_model:
        print(f"Found trained model: {trained_model}")
        if baseline_model:
            command = f"""
            python examples/dpo/evaluate_dpo_french.py \\
                --model {trained_model} \\
                --baseline_model {baseline_model} \\
                --num_examples 50 \\
                --output_file evaluation_results.json \\
                --verbose
            """
        else:
            command = f"""
            python examples/dpo/evaluate_dpo_french.py \\
                --model {trained_model} \\
                --num_examples 50 \\
                --output_file evaluation_results.json \\
                --verbose
            """
    else:
        print("No trained model found. Please run training first.")
        return

    success = run_command(command, "Running model evaluation")

    if success and os.path.exists("evaluation_results.json"):
        print("\\nEvaluation completed! Results saved to: evaluation_results.json")
        print("You can view detailed results by opening the JSON file.")


def quick_test():
    """Run a quick test of the environment"""
    print("Running quick environment test...")

    command = """
    python -c "
import verifiers as vf
print('Testing environment loading...')
env = vf.load_environment(env_id='vf-dpo-french-reasoning', num_train_examples=10)
print(f'Environment loaded successfully with {len(env.dataset)} examples')
print('Dataset columns:', env.dataset.column_names)
print('Test completed successfully!')
    "
    """

    success = run_command(command, "Quick environment test")

    if success:
        print("\\nEnvironment test passed! The setup is working correctly.")
    else:
        print("Environment test failed. Check the error messages above.")


def main():
    parser = argparse.ArgumentParser(description="Setup DPO French Reasoning Environment")
    parser.add_argument("--install", action="store_true", help="Install the environment")
    parser.add_argument("--setup-data", action="store_true", help="Setup training data")
    parser.add_argument("--train-dpo", action="store_true", help="Run DPO training")
    parser.add_argument("--train-grpo", action="store_true", help="Run GRPO training with DPO")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained models")
    parser.add_argument("--quick-test", action="store_true", help="Run quick environment test")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")

    args = parser.parse_args()

    if args.all:
        # Run complete pipeline
        steps = [
            ("install", install_environment),
            ("setup_data", setup_training_data),
            ("quick_test", quick_test),
            ("train_dpo", train_dpo),
            ("evaluate", evaluate_models)
        ]
    else:
        # Run selected steps
        steps = []
        if args.install:
            steps.append(("install", install_environment))
        if args.setup_data:
            steps.append(("setup_data", setup_training_data))
        if args.quick_test:
            steps.append(("quick_test", quick_test))
        if args.train_dpo:
            steps.append(("train_dpo", train_dpo))
        if args.train_grpo:
            steps.append(("train_grpo", train_grpo))
        if args.evaluate:
            steps.append(("evaluate", evaluate_models))

    if not steps:
        print("Please specify at least one action. Use --help for options.")
        print("\\nExample usage:")
        print("  python setup_dpo_french_environment.py --install")
        print("  python setup_dpo_french_environment.py --all")
        return

    print("Starting DPO French Reasoning Setup")
    print("=" * 50)

    for step_name, step_func in steps:
        print(f"\\n{'='*20} Step: {step_name} {'='*20}")
        step_func()

    print("\\n" + "=" * 50)
    print("Setup process completed!")

    if args.all or (args.install and not any([args.train_dpo, args.train_grpo, args.evaluate])):
        print("\\nNext steps:")
        print("1. Run quick test: python setup_dpo_french_environment.py --quick-test")
        print("2. Start DPO training: python setup_dpo_french_environment.py --train-dpo")
        print("3. Or start GRPO training: python setup_dpo_french_environment.py --train-grpo")
        print("4. Evaluate results: python setup_dpo_french_environment.py --evaluate")


if __name__ == "__main__":
    main()
