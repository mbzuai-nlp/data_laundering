import torch
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    GPT2Config
)
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from tqdm import tqdm
import random
import numpy as np
import argparse
import os
import json
import logging
from datetime import datetime

def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    """Compute accuracy metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(predictions)
    print(labels)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

def mmlu_preprocess_function(examples, tokenizer, num_choices=4, max_length=512):
    """Preprocess the dataset for multiple-choice tasks."""
    if isinstance(examples["question"], str):
        examples = {k: [v] for k, v in examples.items()}
    
    batch_size = len(examples["question"])
    
    all_texts = []
    all_labels = []
    for i in range(batch_size):
        question = examples["question"][i]
        choices = examples["choices"][i]
        correct_answer = examples["answer"][i]
        
        # Create question-choice pairs
        text = f"{question} Choices: A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}"
        all_texts.append(text)
        all_labels.append(correct_answer)
        data_save = {'text': text, 'label': correct_answer}
        with open('mmlu_data.jsonl', 'a') as f:
            json.dump(data_save, f)
            f.write('\n')  
    # Tokenize and prepare input
    encoding = tokenizer(all_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(all_labels, dtype=torch.long)
    }

def gpqa_preprocess_function(examples, tokenizer, num_choices=4, max_length=512):
    """
    Preprocess the dataset for multiple-choice tasks by creating separate
    inputs for each choice and assigning labels accordingly.
    """
    questions = examples["Question"]
    batch_size = len(questions)
    
    all_texts = []
    all_labels = []
    
    for i in range(batch_size):
        choices = [
            examples["Incorrect Answer 1"][i],
            examples["Incorrect Answer 2"][i],
            examples["Incorrect Answer 3"][i],
            examples["Correct Answer"][i]
        ]
        
        # Shuffle choices and keep track of correct answer
        correct_answer = choices[3]
        random.shuffle(choices)
        answer = choices.index(correct_answer)
        
        # Combine question and choices
        text = f"{questions[i]} Choices: A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}"
        all_texts.append(text)
        all_labels.append(answer)
        data_save = {'text': text, 'label': answer}
        with open('gpqa_data_12.jsonl', 'a') as f:
            json.dump(data_save, f)
            f.write('\n')  
    # Tokenize and prepare input
    encoding = tokenizer(all_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(all_labels, dtype=torch.long)
    }

def preprocess_function(examples, tokenizer, dataset_name, num_choices=4, max_length=512):
    if "gpqa" in dataset_name.lower():
        return gpqa_preprocess_function(examples, tokenizer, num_choices, max_length)
    elif "mmlu" in dataset_name.lower():
        return mmlu_preprocess_function(examples, tokenizer, num_choices, max_length)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
def setup_logging(experiment_dir):
    """Configure logging to file and console."""
    log_file = os.path.join(experiment_dir, "experiment.log")
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train_teacher_model(args, experiment_dir):
    """Train the teacher model with the specified configurations."""
    # Set seeds for reproducibility
    set_seed(args.seed)

    # Setup logging
    setup_logging(experiment_dir)
    logging.info("Starting teacher model training with the following configuration:")
    logging.info(json.dumps(vars(args), indent=4))

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Configure model
    logging.info("Configuring teacher model...")
    config = GPT2Config.from_pretrained(args.model_name)
    config.num_hidden_layers = args.num_hidden_layers
    config.num_labels = args.num_choices  # Set number of labels to number of choices
    config.pad_token_id = config.eos_token_id  # GPT-2 does not have a pad token by default

    teacher_model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name,
        config=config
    )
    teacher_tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    
    # Add padding token if not present
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        teacher_model.resize_token_embeddings(len(teacher_tokenizer))

    # Load and preprocess dataset
    logging.info(f"Loading dataset: {args.dataset_name}, config: {args.dataset_config}")
    config_names = get_dataset_config_names(args.dataset_name)
    # edinburgh-dawg/mmlu-redux
    if 'all' == args.dataset_config:
            all_congig_datasets = []
            for config_name in tqdm(config_names):
                all_congig_datasets.append(load_dataset(args.dataset_name, config_name, split=args.dataset_split, token=args.hf_token))
            full_dataset = concatenate_datasets(all_congig_datasets)
    else:
        full_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, token=args.hf_token)

    if args.data_size:
        if args.data_size < len(full_dataset):
            full_dataset = full_dataset.shuffle(seed=args.seed).select(range(args.data_size))
            logging.info(f"Selected {args.data_size} samples for training.")
        else:
            logging.info(f"Requested data size {args.data_size} exceeds available data. Using all available data.")

    train_dataset = full_dataset
    eval_dataset = full_dataset  # Modify if you have separate eval data

    logging.info("Preprocessing training dataset...")
    train_encoded = train_dataset.map(
        lambda examples: preprocess_function(examples, teacher_tokenizer, args.train_dataset_name, num_choices=args.num_choices),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_encoded = train_encoded
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=experiment_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        report_to="none"  # Disable default reporting to avoid duplication
    )

    # Save configuration
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Configuration saved to {config_path}")

    # Initialize Trainer
    trainer = Trainer(
        model=teacher_model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")

    # Evaluate the model
    logging.info("Evaluating the trained model...")
    final_results = trainer.evaluate()
    logging.info(f"Final Evaluation Results: {final_results}")

    # Save the model and tokenizer
    logging.info(f"Saving the teacher model to {args.save_model_path}...")
    trainer.save_model(args.save_model_path)
    teacher_tokenizer.save_pretrained(args.save_model_path)
    logging.info("Model and tokenizer saved.")

    # Save training metrics
    metrics_path = os.path.join(experiment_dir, "evaluation_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    logging.info(f"Evaluation results saved to {metrics_path}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Teacher Model for Multiple Choice Tasks")

    # Seed and reproducibility
    parser.add_argument("--seed", type=int, default=321, help="Random seed for reproducibility")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="gpt2", help="Pretrained model name or path")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers in the model")
    parser.add_argument("--num_choices", type=int, default=4, help="Number of choices per question")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="Idavidrein/gpqa", help="Name of the dataset to use")
    parser.add_argument("--dataset_config", type=str, default="gpqa_diamond", help="Configuration of the dataset")
    parser.add_argument("--dataset_split", type=str, default="train", help="Split of the dataset to use")
    parser.add_argument("--data_size", type=int, default=None, help="Number of samples to use for training (default: all)")
    parser.add_argument("--hf_token", type=str, default="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", help="Token for accessing the dataset if required")
    # Training parameters
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy to adopt during training")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Save strategy to adopt during training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size per device for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for the optimizer")
    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps")

    # Model saving parameters
    parser.add_argument("--load_best_model_at_end", action='store_true', help="Load the best model when finished training")
    parser.add_argument("--metric_for_best_model", type=str, default="accuracy", help="Metric to use to compare two different models")
    parser.add_argument("--save_model_path", type=str, default="./saved_teacher_model", help="Path to save the trained model")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./teacher_results", help="Directory to save training outputs")

    return parser.parse_args()

def main():
    """Main function to execute training."""
    args = parse_args()

    # Create a unique experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Update output_dir in args to the experiment_dir
    args.output_dir = experiment_dir

    # Train the teacher model
    train_teacher_model(args, experiment_dir)

if __name__ == "__main__":
    main()
