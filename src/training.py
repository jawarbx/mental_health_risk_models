"""Dataset creation script"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Sequence, Value
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

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
            "MODEL_DIR": MODEL_DIR,
        }.items()
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    cls_report = classification_report(
        labels,
        preds,
        output_dict=False,
        zero_division=0,
    )

    print(cls_report)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="micro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0,
        )
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    try:
        roc_auc_weighted = roc_auc_score(labels, probs, average="weighted")
    except ValueError:
        roc_auc_weighted = float("nan")
    return {
        # macro-averaged
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        # micro-averaged
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        # weighted-averaged
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        # ROC AUC weighted-averaged
        "roc_auc_weighted": roc_auc_weighted,
        # classification report
        "classification_report": cls_report,
    }



def filter_tokenized(batch):
    """Filter samples longer than some threshold"""
    return [len(ids) <= 4096 for ids in batch["input_ids"]]


def main(
    train_split,
    test_split,
    val_split,
    model_output_dir,
    per_device_batch_size,
    num_epochs,
    use_bf16,
    learning_rate,
    use_lora,
):
    """
    Training and testing method
    """
    dataset_dir = f"{OUTPUT_DIR}/{DATASET_DIR}"
    model_output_dir = (
        f"{OUTPUT_DIR}/{MODEL_DIR}" if not model_output_dir else model_output_dir
    )

    assert train_split + test_split + val_split == 1, "Check your splits"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = f"{model_output_dir}/runs/experiment_{timestamp}"

    ensure_dir(model_output_dir)
    ensure_dir(tensorboard_log_dir)

    print("Output directories created:")
    print(f"  - Model: {model_output_dir}")
    print(f"  - TensorBoard logs: {tensorboard_log_dir}")

    dataset = Dataset.load_from_disk(dataset_dir)
    dataset = dataset.cast_column("labels", Sequence(Value("float32")))
    print(f"Original length: {len(dataset)}")
    dataset = dataset.filter(
        filter_tokenized, batched=True, num_proc=4, batch_size=10000
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
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

    num_labels = len(dataset["labels"][0])

    # Free memory from data pipeline
    del dataset, train_temp, train_val

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    if use_lora:
        print("Using lora")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # sequence classification
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["attn.Wqkv"],
        )
        model = get_peft_model(model, lora_config)

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
        metric_for_best_model="f1_micro",
        logging_dir=tensorboard_log_dir,
        logging_first_step=True,
        report_to="tensorboard",
        logging_steps=50,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        bf16=use_bf16,
        gradient_accumulation_steps=4,
        optim="adamw_torch_fused",
        warmup_ratio=0.1,
        tf32=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
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
    parser.add_argument("--per_device_batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_output_dir", type=str, default=None)
    parser.add_argument("--use_bf16", type=bool, default=True)
    parser.add_argument("--use_lora", type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = main(
        train_split=args.train_split,
        test_split=args.test_split,
        val_split=args.val_split,
        model_output_dir=args.model_output_dir,
        per_device_batch_size=args.per_device_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_bf16=args.use_bf16,
        use_lora=args.use_lora,
    )

    if results and results[2] is not None:
        print(results[2])
