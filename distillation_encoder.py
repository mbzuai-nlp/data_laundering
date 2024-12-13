import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertConfig
)
from datasets import load_dataset, DatasetDict, get_dataset_config_names, concatenate_datasets
import torch.nn.functional as F
import random
import numpy as np
import argparse
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm

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
    teacher_model = AutoModelForMultipleChoice.from_pretrained(model_path).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return teacher_model, teacher_tokenizer

def kd_loss(student_logits, teacher_logits, temperature):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return loss

def medmcqa_preprocess_function(examples, tokenizer, num_choices=4):
    first_sentences = [[context] * num_choices for context in examples["question"]]
    second_sentences = [
        [examples["opa"][i], examples["opb"][i], examples["opc"][i], examples["opd"][i]]
        for i in range(len(examples["question"]))
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    return {k: v.view(len(examples["question"]), num_choices, -1) for k, v in tokenized.items()}
    
def mmlu_preprocess_function(examples, tokenizer, num_choices=4):
    """Preprocess the dataset for multiple-choice tasks."""
    if isinstance(examples["question"], str):
        examples = {k: [v] for k, v in examples.items()}
    
    batch_size = len(examples["question"])
    
    # Prepare first sentences (questions)
    first_sentences = []
    second_sentences = []
    labels = []
    
    for i in range(batch_size):
        question = examples["question"][i]
        choices = examples["choices"][i]
        correct_answer = examples["answer"][i]
        
        # Create question-choice pairs
        first_sentences.extend([question] * num_choices)
        second_sentences.extend(choices)
        labels.append(correct_answer)
    
    # Tokenize inputs
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Reshape tensors to [batch_size, num_choices, seq_length]
    tokenized_examples = {k: v.view(batch_size, num_choices, -1) for k, v in tokenized_examples.items()}
    tokenized_examples["labels"] = torch.tensor(labels)
    
    return tokenized_examples

def gpqa_preprocess_function(examples, tokenizer, num_choices=4):
    if isinstance(examples["Question"], str):
        examples = {k: [v] for k, v in examples.items()}
    
    batch_size = len(examples["Question"])
    first_sentences = [[question] * num_choices for question in examples["Question"]]
    
    question_headers = examples.get("question_header", [""] * batch_size)
    second_sentences = []
    labels = []
    
    for i, header in enumerate(question_headers):
        choices = [
            examples["Incorrect Answer 1"][i],
            examples["Incorrect Answer 2"][i],
            examples["Incorrect Answer 3"][i],
            examples["Correct Answer"][i]
        ]
        correct_answer = examples["Correct Answer"][i]
        
        shuffled_choices = choices.copy()
        random.shuffle(shuffled_choices)
        
        label = shuffled_choices.index(correct_answer)
        labels.append(label)
        
        second_sentences.append([f"{choice}" for choice in shuffled_choices])
    
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized = {k: v.view(batch_size, num_choices, -1) for k, v in tokenized.items()}
    tokenized["labels"] = torch.tensor(labels)
    
    return tokenized

def race_preprocess_function(examples, tokenizer, num_choices=4):
    first_sentences = [[context] * num_choices for context in examples["article"]]
    second_sentences = []
    
    for question, options in zip(examples["question"], examples["options"]):
        second_sentences.append([f"{question} {opt}" for opt in options])
    
    # Flatten lists
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Convert letter labels (A,B,C,D) to numeric indices (0,1,2,3)
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    labels = [label_map[label] for label in examples["answer"]]
    
    return {
        **{k: v.view(-1, num_choices, v.size(-1)) for k, v in tokenized.items()},
        "labels": torch.tensor(labels)
    }

def preprocess_function(examples, tokenizer, dataset_name):
    if "race" in dataset_name.lower():
        return race_preprocess_function(examples, tokenizer)
    elif "medmcqa" in dataset_name.lower():
        return medmcqa_preprocess_function(examples, tokenizer)
    elif "gpqa" in dataset_name.lower():
        return gpqa_preprocess_function(examples, tokenizer)
    elif "mmlu" in dataset_name.lower():
        return mmlu_preprocess_function(examples, tokenizer)
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
        outputs = model(**inputs)
        student_logits = outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

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

    # Load student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)

    # Configure student model
    logging.info("Configuring student model...")
    config = BertConfig.from_pretrained(args.student_model_name)
    config.num_hidden_layers = args.student_num_layers

    student_model = AutoModelForMultipleChoice.from_pretrained(
        args.student_model_name,
        config=config
    )

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
    config_names = get_dataset_config_names(args.eval_dataset_name)
        # edinburgh-dawg/mmlu-redux
    if 'all' == args.eval_dataset_config:
        all_congig_datasets = []
        for config_name in tqdm(config_names):
            all_congig_datasets.append(load_dataset(args.eval_dataset_name, config_name, split=args.eval_dataset_split, token=args.hf_token))
        eval_dataset = concatenate_datasets(all_congig_datasets)
    else:
        eval_dataset = load_dataset(args.eval_dataset_name, args.eval_dataset_config, split=args.eval_dataset_split, token=args.hf_token)
    
    eval_encoded = eval_dataset.map(
        lambda examples: preprocess_function(examples, student_tokenizer, args.eval_dataset_name),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    logging.info("Evaluation dataset encoded.")

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
        load_best_model_at_end=args.load_best_model_at_end,
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
    eval_datasets = DatasetDict({
        "train": train_encoded,
        "eval": eval_encoded
    })
    # Initialize Trainer with distillation
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
        eval_dataset=eval_datasets, 
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
    logging.info("Loading and preprocessing training dataset...")
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
    eval_dataset = load_dataset(
        args.eval_dataset_name,
        args.eval_dataset_config,
        split=args.eval_dataset_split,
        token=args.hf_token
    )
    eval_encoded = eval_dataset.map(
        lambda examples: preprocess_function(examples, student_tokenizer, args.eval_dataset_name),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    logging.info("Evaluation dataset encoded.")
    eval_datasets = DatasetDict({
        "train": train_encoded,
        "eval": eval_encoded
    })
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
    test_results = eval_trainer.evaluate(eval_dataset=eval_datasets)
    logging.info(f"Evaluation results: {test_results}")

    return test_results

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Multiple Choice Models")

    # Seed and device
    parser.add_argument("--seed", type=int, default=321, help="Random seed for reproducibility")

    # Data parameters
    parser.add_argument("--data_size", type=int, default=20000, help="Number of samples to use for training")

    # Teacher model parameters
    parser.add_argument("--teacher_model_path", type=str, default="./saved_teacher_model", help="Path or name of the teacher model")
    parser.add_argument("--teacher_tokenizer_path", type=str, default="./saved_teacher_model", help="Path or name of the teacher tokenizer")

    # Student model parameters
    parser.add_argument("--student_model_name", type=str, default="bert-base-uncased", help="Name of the student model")
    parser.add_argument("--student_num_layers", type=int, default=2, help="Number of hidden layers in the student model")

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
    parser.add_argument("--hf_token", type=str, default="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", help="Token for accessing the evaluation dataset if required")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./student_results", help="Directory to save training outputs")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Evaluation batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--load_best_model_at_end", action='store_true', help="Load the best model at the end of training")
    parser.add_argument("--metric_for_best_model", type=str, default="accuracy", help="Metric to use for selecting the best model")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for optimizer")

    # Save model parameters
    parser.add_argument("--save_model_path", type=str, default="./saved_student_model", help="Path to save the student model")
    parser.add_argument("--eval_only", action='store_true', help="Perform evaluation only, without training")

    return parser.parse_args()

def main():
    args = parse_args()

    # Create a unique experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Update output_dir in args to the experiment_dir
    args.output_dir = experiment_dir
    if args.eval_only:
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

        logging.info("Starting Evaluation:")
        student_model, student_tokenizer = load_teacher_model("./saved_student_model", "./teacher_results/experiment_20240929_175242/saved_teacher_model", 'cuda')
        test_results = test_student_on_teacher_benchmark(
            student_model,
            student_tokenizer,
            args,
            experiment_dir
        )
        eval_results_path=os.path.join(experiment_dir, 'evaluation_results.jsonl')
        with open(eval_results_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        logging.info(f"Evaluation results saved to {eval_results_path}.")
    else:
        train_student_model(args, experiment_dir)
        

if __name__ == "__main__":
    main()