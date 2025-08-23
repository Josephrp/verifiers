import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset, load_dataset, concatenate_datasets
import verifiers as vf
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import torch


class FrenchThinkingGenerator:
    """Generate French thinking traces when missing"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.translator = GoogleTranslator(source='auto', target='fr')

    def load_model(self):
        """Load the local model for thinking generation"""
        if self.model is None:
            print(f"Loading thinking generation model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_thinking(self, prompt: str, max_length: int = 200) -> str:
        """Generate thinking trace for a given prompt"""
        self.load_model()

        thinking_prompt = f"Réfléchissons étape par étape pour répondre à cette question:\n{prompt}\n\nRaisonnement:"
        inputs = self.tokenizer(thinking_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        thinking = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return thinking.strip()

    def is_french(self, text: str) -> bool:
        """Check if text is in French"""
        try:
            return detect(text) == 'fr'
        except:
            return False

    def translate_to_french(self, text: str) -> str:
        """Translate text to French"""
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"Translation failed: {e}")
            return text


class DPOFrenchReasoningEnvironment(vf.MultiTurnEnv):
    """Multi-turn environment for DPO training with French reasoning"""

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        thinking_generator: Optional[FrenchThinkingGenerator] = None,
        max_turns: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.thinking_generator = thinking_generator or FrenchThinkingGenerator()
        self.max_turns = max_turns

        # Create rubric for evaluation
        self.rubric = vf.Rubric(
            funcs=[self._preference_reward, self._french_quality_reward],
            weights=[1.0, 0.5]
        )

    def _preference_reward(self, prompt, completion, answer, state, **kwargs) -> float:
        """Reward based on preference between accepted/rejected responses"""
        if 'accepted' not in state or 'rejected' not in state:
            return 0.0

        # Simple preference-based reward
        # In a real implementation, you'd use a preference model
        response = completion[-1]['content'] if completion else ""

        # Check if response aligns with accepted pattern
        accepted_keywords = self._extract_keywords(state['accepted'])
        rejected_keywords = self._extract_keywords(state['rejected'])

        accepted_score = sum(1 for keyword in accepted_keywords if keyword.lower() in response.lower())
        rejected_score = sum(1 for keyword in rejected_keywords if keyword.lower() in response.lower())

        return max(0.0, accepted_score - rejected_score)

    def _french_quality_reward(self, prompt, completion, answer, state, **kwargs) -> float:
        """Reward for French language quality and thinking structure"""
        reward = 0.0

        for message in completion:
            if message['role'] == 'assistant':
                content = message['content']

                # Check if response contains French thinking
                if self.thinking_generator.is_french(content):
                    reward += 1.0

                # Check for structured thinking patterns
                if any(word in content.lower() for word in ['raisonnons', 'réfléchissons', 'étape', 'premièrement']):
                    reward += 0.5

        return reward

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for preference analysis"""
        # Simple keyword extraction - in practice, you'd use more sophisticated methods
        words = text.split()
        return [word.strip('.,!?') for word in words if len(word) > 3]

    def is_completed(self, messages: List[Dict], state: Dict, **kwargs) -> bool:
        """Check if the interaction is completed"""
        turn_count = state.get('turn_count', 0)
        return turn_count >= self.max_turns or state.get('completed', False)

    def env_response(self, messages: List[Dict], state: Dict, **kwargs) -> Tuple[List[Dict], Dict]:
        """Generate environment response"""
        turn_count = state.get('turn_count', 0)

        if turn_count == 0:
            # First turn - provide thinking guidance if needed
            prompt = state.get('prompt', '')
            thinking = state.get('thinking', '')

            if not thinking or not self.thinking_generator.is_french(thinking):
                if not thinking:
                    thinking = self.thinking_generator.generate_thinking(prompt)
                if not self.thinking_generator.is_french(thinking):
                    thinking = self.thinking_generator.translate_to_french(thinking)

                state['thinking'] = thinking

            response = [{
                "role": "user",
                "content": f"Voici un raisonnement en français pour vous aider:\n\n{thinking}\n\nMaintenant, fournissez votre réponse:"
            }]

        else:
            # Subsequent turns - provide feedback
            response = [{
                "role": "user",
                "content": "Continuez votre raisonnement en français."
            }]

        state['turn_count'] = turn_count + 1
        return response, state


def load_multiple_datasets(dataset_names: List[str], splits: List[str] = None) -> Dataset:
    """Load and combine multiple datasets"""
    if splits is None:
        splits = ['train'] * len(dataset_names)

    datasets = []
    for name, split in zip(dataset_names, splits):
        try:
            # Try loading from Hugging Face
            dataset = load_dataset(name, split=split)
            datasets.append(dataset)
        except Exception as e:
            print(f"Failed to load dataset {name}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets could be loaded")

    # Combine datasets
    combined_dataset = concatenate_datasets(datasets)

    return combined_dataset


def preprocess_for_dpo(dataset: Dataset, thinking_generator: FrenchThinkingGenerator) -> Dataset:
    """Preprocess dataset into DPO format with 4 columns: prompt, thinking, accepted, rejected"""

    def process_example(example):
        prompt = example.get('prompt') or example.get('question', '')
        answer = example.get('answer', '')
        info = example.get('info', {})

        # Extract thinking if available, otherwise generate
        thinking = example.get('thinking') or info.get('thinking', '')
        if not thinking:
            thinking = thinking_generator.generate_thinking(prompt)

        # Ensure thinking is in French
        if not thinking_generator.is_french(thinking):
            thinking = thinking_generator.translate_to_french(thinking)

        # Generate accepted and rejected responses
        # In practice, you might have these from human preferences or other sources
        accepted = answer  # Assume the ground truth is the accepted answer

        # Generate a rejected answer (simple heuristic - make it incorrect)
        rejected = generate_rejected_answer(answer)

        return {
            'prompt': prompt,
            'thinking': thinking,
            'accepted': accepted,
            'rejected': rejected,
            'original_answer': answer
        }

    def generate_rejected_answer(correct_answer: str) -> str:
        """Generate a plausible but incorrect answer for DPO training"""
        # Simple rejection strategy - modify the correct answer
        if correct_answer.isdigit():
            # For numerical answers, provide a wrong number
            num = int(correct_answer)
            rejected = str(num + 1 if num < 100 else num - 1)
        else:
            # For text answers, provide a modified version
            words = correct_answer.split()
            if len(words) > 1:
                rejected = f"{' '.join(words[:-1])} incorrect"
            else:
                rejected = f"{correct_answer} (incorrect)"

        return rejected

    # Process the dataset
    processed_dataset = dataset.map(process_example)

    # Filter to ensure we have all required columns
    required_columns = ['prompt', 'thinking', 'accepted', 'rejected']
    available_columns = [col for col in required_columns if col in processed_dataset.column_names]

    if len(available_columns) < len(required_columns):
        missing = set(required_columns) - set(available_columns)
        raise ValueError(f"Missing required columns: {missing}")

    return processed_dataset


def load_environment(
    dataset_names: List[str] = None,
    local_model: str = "microsoft/DialoGPT-medium",
    max_turns: int = 5,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    **kwargs
) -> DPOFrenchReasoningEnvironment:
    """
    Load the DPO French reasoning environment

    Args:
        dataset_names: List of dataset names to load
        local_model: Local model for thinking generation
        max_turns: Maximum turns per interaction
        num_train_examples: Number of training examples (-1 for all)
        num_eval_examples: Number of eval examples (-1 for all)
    """

    if dataset_names is None:
        dataset_names = ["gsm8k", "math_dataset"]  # Default datasets

    # Initialize thinking generator
    thinking_generator = FrenchThinkingGenerator(model_name=local_model)

    # Load and combine datasets
    print(f"Loading datasets: {dataset_names}")
    raw_dataset = load_multiple_datasets(dataset_names)

    # Preprocess for DPO format
    print("Preprocessing dataset for DPO format...")
    dataset = preprocess_for_dpo(raw_dataset, thinking_generator)

    # Create eval dataset
    eval_dataset = None
    if num_eval_examples != -1:
        eval_dataset = dataset.select(range(min(num_eval_examples, len(dataset))))
    elif len(dataset) > 1000:
        # Use last 1000 examples for eval
        eval_dataset = dataset.select(range(len(dataset) - 1000, len(dataset)))
        dataset = dataset.select(range(len(dataset) - 1000))

    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    print(f"Train dataset size: {len(dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")

    # Create environment
    env = DPOFrenchReasoningEnvironment(
        dataset=dataset,
        eval_dataset=eval_dataset,
        thinking_generator=thinking_generator,
        max_turns=max_turns,
        **kwargs
    )

    return env


# Example usage and testing functions
def create_sample_dpo_data():
    """Create sample data for testing"""
    sample_data = [
        {
            'prompt': 'Combien font 2 + 2?',
            'answer': '4',
            'thinking': 'Addition simple: 2 plus 2 égale 4.'
        },
        {
            'prompt': 'Quelle est la capitale de la France?',
            'answer': 'Paris',
            'thinking': 'La France est un pays d\'Europe. Sa capitale historique et actuelle est Paris.'
        }
    ]

    return Dataset.from_list(sample_data)


if __name__ == "__main__":
    # Test the environment
    env = load_environment(
        dataset_names=["gsm8k"],  # Use GSM8K as test
        num_train_examples=10,
        num_eval_examples=5
    )
    print("Environment loaded successfully!")
    print(f"Dataset columns: {env.dataset.column_names}")
