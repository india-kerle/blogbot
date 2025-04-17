from blogbot import PROJECT_DIR

from datasets import Dataset

import logging

from utils import training_image
from utils import VOLUME_CONFIG

from blogbot.configs import ModelTrainingConfig

from modal import Secret
from modal import App

app = App(
    name="blogbot-training",
    image=training_image,
    secrets=[Secret.from_name("huggingface-token")],
)

synthetic_data_path = PROJECT_DIR / "data/processed"
CONTAINER_DATA_ROOT = "/data"

model_training_config = ModelTrainingConfig().model_dump()

@app.function(
    volumes=VOLUME_CONFIG
)
def process_data() -> tuple[Dataset, Dataset]:
    """Process data for training."""
    import os
    import datasets
    from transformers import AutoTokenizer

    data_path = os.path.join('/data', "synthetic_data.parquet")

    raw_dataset = datasets.load_dataset("parquet", data_files=data_path)
    hf_dataset= raw_dataset['train']

    logging.info(f"Raw dataset loaded with {len(hf_dataset)} samples.")
    
    split_dataset_dict = hf_dataset.train_test_split(test_size=model_training_config['test_size'], seed=model_training_config['random_state'])
    tokenizer = AutoTokenizer.from_pretrained(model_training_config['model_name'])
        
    def preprocess_function(examples):
        return tokenizer(examples["value"], truncation=True, padding="max_length", max_length=model_training_config['max_length'])

    tokenized_datasets = split_dataset_dict.map(preprocess_function, batched=True)

    # Remove columns not needed by the model
    tokenized_datasets = tokenized_datasets.remove_columns(["value", "id"]) # Remove original text and id
    # Ensure the format is set for PyTorch
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    VOLUME_CONFIG['/data'].commit()  # Commit the changes to the volume

    return train_dataset, eval_dataset

@app.function(
    image=training_image,
    mounts=[data_mount],
    volumes=VOLUME_CONFIG
)
def train_model(train_data: Dataset, test_data: Dataset) -> None:
    """Train model using synthetic data."""
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import torch
    import os 

    model = AutoModelForSequenceClassification.from_pretrained(
            model_training_config['model_name'],
            num_labels=model_training_config['num_labels'],
        )
    tokenizer = AutoTokenizer.from_pretrained(model_training_config['model_name'])

    # Create output directory if it doesn't exist
    os.makedirs(model_training_config['output_dir'], exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=model_training_config['output_dir'],
        num_train_epochs=model_training_config['num_train_epochs'],
        learning_rate=model_training_config['learning_rate'],
        per_device_train_batch_size=model_training_config['train_batch_size'],
        per_device_eval_batch_size=model_training_config['eval_batch_size'],
        weight_decay=model_training_config['weight_decay'],
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save checkpoint at the end of each epoch
        logging_steps=10,            # Log training loss every 10 steps
        load_best_model_at_end=True, # Load the best model checkpoint at the end
        metric_for_best_model="f1",  # Use F1 score to determine the best model
        greater_is_better=True,
        report_to="none",            # Disable wandb/tensorboard reporting unless configured
        fp16=torch.cuda.is_available(), # Enable mixed-precision training if GPU available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data, # Using the test split as the evaluation set
        tokenizer=tokenizer,
    )

    trainer.train()
    # Save the model
    model_path = os.path.join(model_training_config['output_dir'], "final_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    logging.info(f"Model trained and saved to {model_path}")

    VOLUME_CONFIG['/fine-tuned'].commit()

@app.function(
    image=training_image,
    volumes=VOLUME_CONFIG
)
def evaluate_model(model_path: str, test_data: Dataset) -> dict:
    """Evaluate the trained model on the test dataset."""
    from transformers import pipeline
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import torch
    import numpy as np    
    #in the volume 
    model_path = "/fine-tuned/final_model"

    classifier = pipeline(
        "text-classification", 
        model=model_path, 
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )

    texts = test_data["input_ids"]  # Assuming tokenized inputs
    labels = test_data["labels"]
    
    # Run inference
    predictions = classifier(texts)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics

@app.local_entrypoint
def main() -> None:
    """Main function to run the training pipeline.
    
    modal run --detach blogbot/train.py
    """
    train_data, eval_data = process_data.remote()
    train_model.remote(train_data, eval_data)
    metrics = evaluate_model.remote(eval_data)

    logging.info(f"Training completed! Final evaluation metrics: {metrics}")
    