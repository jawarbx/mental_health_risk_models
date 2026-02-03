"""Dataset creation script"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data_pipeline import DataPipeline
from preprocess import ensure_dir

SCRIPT_DIR = Path(__file__).parent.resolve()
dotenv_path = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

MODEL_NAME = os.getenv("MODEL_NAME")
DATASET_DIR = os.getenv("DATASET_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")

if not all([MODEL_NAME, DATASET_DIR, OUTPUT_DIR, MODEL_DIR]):
    missing = [
        var
        for var, val in {
            "MODEL_NAME": MODEL_NAME,
            "DATASET_DIR": DATASET_DIR,
            "OUTPUT_DIR": OUTPUT_DIR,
            "MODEL_DIR": MODEL_DIR
        }.items()
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main(
    month_deltas: list[int],
    matching_method=None,
    train_split=0.70,
    test_split=0.05,
    val_split=0.25,
    model_output_dir=None,
    per_device_batch_size=16,
    num_epochs=3,
    use_bf16=True,
    learning_rate=2e-5,
    MODEL_DIR=None
):
    """
    Training and testing method
    """
    dataset_dir = f"{OUTPUT_DIR}/{DATASET_DIR}"
    model_output_dir = f"{OUTPUT_DIR}/{MODEL_DIR}" if not model_output_dir else model_output_dir

    assert(train_split + test_split + val_split == 1), "Check your splits"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = f"{model_output_dir}/runs/experiment_{timestamp}"

    ensure_dir(dataset_dir)
    ensure_dir(model_output_dir)
    ensure_dir(tensorboard_log_dir)

    print("Output directories created:")
    print(f"  - Dataset: {dataset_dir}")
    print(f"  - Model: {model_output_dir}")
    print(f"  - TensorBoard logs: {tensorboard_log_dir}")

    dataset = Dataset.load_from_disk(data_output_dir)
    train_temp = dataset.train_test_split(test_size=test_split, seed=42)
    val_size = val_split / (train_split + val_split)
    train_val = train_temp["train"].train_test_split(test_size=val_size, seed=42)

    dataset_dict = DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_temp["test"],
        }
    )

    print(f"Train size: {len(dataset_dict['train'])}")
    print(f"Validation size: {len(dataset_dict['validation'])}")
    print(f"Test size: {len(dataset_dict['test'])}")

    # Free memory from data pipeline
    del dataset, train_temp, train_val

    num_labels = len(dataset["labels"][0])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=tensorboard_log_dir,
        logging_first_step=True,
        report_to="tensorboard",
        logging_steps=50,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        bf16=use_bf16,
        gradient_accumulation_steps=1,
        optim="adamw_torch_fused",
        warmup_ratio=0.1,
        tf32=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(dataset_dict["test"])
    print(f"Test results: {test_results}")

    final_model_path = f"{model_output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}")

    return dataset_dict, trainer, test_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MCI prediction model")
    parser.add_argument("--train_split", type=float, default=0.70)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--val_split", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_output_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = main(
        train_split=args.train_split,
        test_split=args.test_split,
        val_split=args.val_split,
        model_output_dir=args.model_output_dir,
        per_device_batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )

    if results and results[2] is not None:
        print(results[2])
