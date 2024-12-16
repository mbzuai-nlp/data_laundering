import torch
from transformers import (
    GPT2ForSequenceClassification,  
    GPT2Tokenizer,                 
    Trainer,
    TrainingArguments,
    GPT2Config,                     
)
from datasets import load_dataset, DatasetDict
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import argparse
import os
import json
import logging
from datetime import datetime
from datasets import Dataset, DatasetDict

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

def load_teacher_model(model_path, tokenizer_path, device):
    teacher_model = GPT2ForSequenceClassification.from_pretrained(model_path).to(device)
    teacher_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return teacher_model, teacher_tokenizer

def kd_loss(student_logits, teacher_logits, temperature):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return loss

def preprocess_function_benchmark(examples, tokenizer, num_choices=4, max_length=512):
    """
    Preprocess the dataset for multiple-choice tasks by creating separate
    inputs for each choice and assigning labels accordingly.
    """
    
    all_texts = examples["text"]
    all_labels = examples["label"]
    
    # Tokenize and prepare input
    encoding = tokenizer(all_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(all_labels, dtype=torch.long)
    }

def medmcqa_preprocess_function(examples, tokenizer, num_choices=4, max_length=512):
    """
    Preprocess the dataset for multiple-choice tasks by creating separate
    inputs for each choice and assigning labels accordingly.
    """
    questions = examples["question"]
    batch_size = len(questions)
    
    all_texts = []
    all_labels = []
    
    for i in range(batch_size):
        choices = [
            examples["opa"][i],
            examples["opb"][i],
            examples["opc"][i],
            examples["opd"][i],
        ]
        
        # Shuffle choices and keep track of correct answer
        correct_answer = choices[3]
        # random.shuffle(choices)
        answer = choices.index(correct_answer)
        
        # Combine question and choices
        text = f"{questions[i]} Choices: A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}"
        all_texts.append(text)
        all_labels.append(answer)

    # Tokenize and prepare input
    encoding = tokenizer(all_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
    }

def race_preprocess_function(examples, tokenizer, num_choices=4, max_length=512):
    questions = examples["question"]
    batch_size = len(questions)
    
    all_texts = []
    all_labels = []
    
    for i in range(batch_size):
        choices = examples["options"][i]
        
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        answer = label_map[examples["answer"][i]]

        # Combine question and choices
        text = f"{questions[i]} Choices: A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}"
        all_texts.append(text)
        all_labels.append(answer)

    # Tokenize and prepare input
    encoding = tokenizer(all_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(all_labels, dtype=torch.long)
    }

def preprocess_function(examples, tokenizer, dataset_name):
    if "race" in dataset_name.lower():
        return race_preprocess_function(examples, tokenizer)
    elif "medmcqa" in dataset_name.lower():
        return medmcqa_preprocess_function(examples, tokenizer)
    elif "gpqa" or "mmlu" in dataset_name.lower():
        return preprocess_function_benchmark(examples, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature, alpha, loss_fn, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.loss_fn = loss_fn
        self.device = device

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").to(self.device)
        # Reshape inputs for multiple choices
        input_ids = inputs["input_ids"].view(-1, inputs["input_ids"].size(-1))
        attention_mask = inputs["attention_mask"].view(-1, inputs["attention_mask"].size(-1))
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        outputs = model(**inputs)
        student_logits = outputs.logits.view(-1, 4)  # Assuming num_choices=4

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits.view(-1, 4)

        ce_loss = F.cross_entropy(student_logits, labels)

        if self.loss_fn == 'kd':
            distil_loss = kd_loss(student_logits, teacher_logits, self.temperature)
        elif self.loss_fn == 'mse':
            distil_loss = F.mse_loss(student_logits, teacher_logits)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

        loss = self.alpha * distil_loss + (1 - self.alpha) * ce_loss

        return (loss, outputs) if return_outputs else loss

def train_student_model(args, experiment_dir):
    # Set seeds for reproducibility
    set_seed(args.seed)

    # Configure logging
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

    logging.info("Starting training with the following configuration:")
    logging.info(vars(args))

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load teacher model and tokenizer
    logging.info("Loading teacher model and tokenizer...")
    teacher_model, teacher_tokenizer = load_teacher_model(
        model_path=args.teacher_model_path,
        tokenizer_path=args.teacher_tokenizer_path,
        device=device
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        teacher_model.resize_token_embeddings(len(teacher_tokenizer))
        teacher_model.config.pad_token_id = teacher_tokenizer.pad_token_id
    # Load student tokenizer
    student_tokenizer = GPT2Tokenizer.from_pretrained(args.student_model_name)
    # student_tokenizer.pad_token = student_tokenizer.eos_token  # GPT2 doesn't have a pad token by default
    if student_tokenizer.pad_token is None:
        student_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # student_model.resize_token_embeddings(len(student_tokenizer))
        
    # Configure student model
    logging.info("Configuring student model...")
    config = GPT2Config.from_pretrained(args.student_model_name)
    config.num_labels = 4  # Number of choices

    student_model = GPT2ForSequenceClassification.from_pretrained(
        args.student_model_name,
        config=config
    )
    student_model.resize_token_embeddings(len(student_tokenizer))  # Resize embeddings if pad token was added
    student_model.config.pad_token_id = student_tokenizer.pad_token_id
    # Load and preprocess training dataset
    logging.info("Loading and preprocessing training dataset...")
    if "race" in args.train_dataset_name.lower():
        train_dataset = load_dataset(
            args.train_dataset_name,
            args.train_dataset_config,  # 'all', 'high', or 'middle'
            split=args.train_dataset_split
        )
    else:
        train_dataset = load_dataset(
            args.train_dataset_name,
            split=args.train_dataset_split
        )
    if args.data_size < len(train_dataset):
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.data_size))
        logging.info(f"Selected {args.data_size} samples for training.")
    else:
        logging.info(f"Requested data size {args.data_size} exceeds available data. Using all available data.")

    train_encoded = train_dataset.map(
        lambda examples: preprocess_function(examples, student_tokenizer, args.train_dataset_name),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    if "race" in args.train_dataset_name.lower():
        # Labels are already added in race_preprocess_function
        pass
    elif "medmcqa" in args.train_dataset_name.lower():
        if "cop" in train_dataset.column_names:
            labels = train_dataset["cop"]
        elif "label" in train_dataset.column_names:
            labels = train_dataset["label"]
        else:
            raise ValueError("Dataset does not contain 'cop' or 'label' column for labels.")
        train_encoded = train_encoded.add_column("labels", labels)
    logging.info("Training dataset encoded and labels added.")

    # Load and preprocess evaluation dataset
    logging.info("Loading and preprocessing evaluation dataset...")
    path_data = f'gpqa_data_{args.student_num_layers}.jsonl' if 'Idavidrein/gpqa' in args.eval_dataset_name else 'mmlu_data.jsonl'
    eval_dataset = pd.read_json(path_data, lines=True)
    eval_dataset = Dataset.from_pandas(eval_dataset)
    eval_encoded = eval_dataset.map(
        lambda examples: preprocess_function(examples, student_tokenizer, args.eval_dataset_name),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    logging.info("Evaluation dataset encoded.")

    # Create a DatasetDict with both train and eval datasets for evaluation on training data
    eval_datasets = DatasetDict({
        "train": train_encoded,
        "eval": eval_encoded
    })

    # Training arguments
    training_args = TrainingArguments(
        output_dir=experiment_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(experiment_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        report_to="none"  # Disable default reporting to avoid duplication
    )

    # Save the configuration to a JSON file
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Experiment configuration saved to {config_path}.")

    # Initialize Trainer with distillation and multiple eval datasets
    logging.info("Initializing Trainer...")
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=args.temperature,
        alpha=args.alpha,
        loss_fn=args.loss_function,
        device=device,
        model=student_model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=eval_datasets,  # Pass the DatasetDict here
        compute_metrics=compute_metrics
    )

    # Train the student model
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")

    # Save the student model
    logging.info(f"Saving student model to {os.path.join(experiment_dir, args.save_model_path)}...")
    os.makedirs(os.path.join(experiment_dir, args.save_model_path), exist_ok=True)
    trainer.save_model(os.path.join(experiment_dir, args.save_model_path))
    # Save training arguments and results
    training_args_dict = training_args.to_dict()
    with open(os.path.join(experiment_dir, "training_args.json"), 'w') as f:
        json.dump(training_args_dict, f, indent=4)
    logging.info("Training arguments saved.")

    # Save the final training metrics
    with open(os.path.join(experiment_dir, "training_metrics.json"), 'w') as f:
        json.dump(trainer.state.log_history, f, indent=4)
    logging.info("Training metrics saved.")
 
    # Evaluate the student model
    logging.info("Evaluating student model on benchmark dataset...")
    logging.info("Evaluation completed.")
    logging.info("Running evaluation...")
    test_results = trainer.evaluate(eval_dataset=eval_datasets)
    logging.info(f"Evaluation results: {test_results}")
    # Save evaluation results
    eval_results_path = os.path.join(experiment_dir, "evaluation_results.json")
    with open(eval_results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    logging.info(f"Evaluation results saved to {eval_results_path}.")

def test_student_on_teacher_benchmark(student_model, student_tokenizer, args, experiment_dir):
    # Load benchmark dataset used by the teacher
    logging.info("Loading benchmark dataset for evaluation...")
    eval_dataset = pd.read_json('gpqa_data.jsonl', lines=True)
    eval_dataset = Dataset.from_pandas(eval_dataset)
    logging.info("Preprocessing benchmark dataset...")
    benchmark_encoded = eval_dataset.map(
        lambda examples: preprocess_function(examples, student_tokenizer, args.eval_dataset_name),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Initialize Trainer for evaluation
    logging.info("Initializing Trainer for evaluation...")
    eval_args = TrainingArguments(
        output_dir=os.path.join(experiment_dir, "eval"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=os.path.join(experiment_dir, "logs"),
        logging_steps=10,
        report_to="none"  # Disable default reporting
    )
    eval_trainer = Trainer(
        model=student_model,
        args=eval_args,
        compute_metrics=compute_metrics
    )

    # Evaluate
    logging.info("Running evaluation...")
    test_results = eval_trainer.evaluate(eval_dataset=benchmark_encoded)
    logging.info(f"Evaluation results: {test_results}")

    return test_results

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Multiple Choice Models using GPT-2")

    # Seed and device
    parser.add_argument("--seed", type=int, default=321, help="Random seed for reproducibility")

    # Data parameters
    parser.add_argument("--data_size", type=int, default=20000, help="Number of samples to use for training")

    # Teacher model parameters
    parser.add_argument("--teacher_model_path", type=str, default="./saved_teacher_model", help="Path or name of the teacher model")
    parser.add_argument("--teacher_tokenizer_path", type=str, default="./saved_teacher_model", help="Path or name of the teacher tokenizer")

    # Student model parameters
    parser.add_argument("--student_model_name", type=str, default="gpt2", help="Name of the student model")
    parser.add_argument("--student_num_layers", type=int, default=12, help="Number of hidden layers in the student model")  # GPT-2 base has 12 layers

    # Loss parameters
    parser.add_argument("--loss_function", type=str, choices=['kd', 'mse'], default='kd', help="Loss function to use for distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for the distillation loss")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")

    # Training dataset parameters
    parser.add_argument("--train_dataset_name", type=str, default="openlifescienceai/medmcqa", help="Name of the training dataset")
    parser.add_argument("--train_dataset_config", type=str, default="all", help="Configuration of the training dataset")
    parser.add_argument("--train_dataset_split", type=str, default="train", help="Split of the training dataset to use")

    # Evaluation dataset parameters
    parser.add_argument("--eval_dataset_name", type=str, default="Idavidrein/gpqa", help="Name of the evaluation dataset")
    parser.add_argument("--eval_dataset_config", type=str, default="gpqa_diamond", help="Configuration of the evaluation dataset")
    parser.add_argument("--eval_dataset_split", type=str, default="train", help="Split of the evaluation dataset to use")
    parser.add_argument("--eval_dataset_token", type=str, default="", help="Token for accessing the evaluation dataset if required")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./student_results", help="Directory to save training outputs")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size per device")  # Reduced due to GPT-2's memory usage
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--metric_for_best_model", type=str, default="accuracy", help="Metric to use for selecting the best model")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")  # Lower learning rate for GPT-2

    # Save model parameters
    parser.add_argument("--save_model_path", type=str, default="./saved_student_model", help="Path to save the student model")

    return parser.parse_args()

def main():
    args = parse_args()

    # Create a unique experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Update output_dir in args to the experiment_dir
    args.output_dir = experiment_dir

    train_student_model(args, experiment_dir)

if __name__ == "__main__":
    main()
