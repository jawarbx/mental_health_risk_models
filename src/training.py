"""Model Training script"""

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
from sklearn.model_selection import train_test_split
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
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mci-prediction")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", None)
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
    preds = (probs >= 0.15).astype(int)

    cls_report = classification_report(
        labels,
        preds,
        output_dict=False,
        zero_division=0,
    )

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
    return [len(ids) <= 8192 for ids in batch["input_ids"]]


def split_dataset_by_id(
    dataset: Dataset,
    id_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int = 42,
) -> DatasetDict:
    """
    Split a HuggingFace Dataset by unique IDs to prevent data leakage.
    All rows sharing the same ID are guaranteed to land in the same split.

    Args:
    dataset:    A HuggingFace Dataset (already filtered / cast).
    id_column:  Name of the column that carries the group identifier.
    train_size: Fraction of unique IDs assigned to train  (e.g. 0.70).
    val_size:   Fraction of unique IDs assigned to val    (e.g. 0.25).
    test_size:  Fraction of unique IDs assigned to test   (e.g. 0.05).
    seed:       Random seed for reproducibility.

    Returns:
    DatasetDict with keys "train", "validation", "test".
    """
    assert (
        abs(train_size + val_size + test_size - 1.0) < 1e-9
    ), "train_size + val_size + test_size must equal 1.0"

    unique_ids = sorted(set(dataset[id_column]))
    print(f"\n[ID Split] Unique IDs : {len(unique_ids)}")
    print(f"[ID Split] Total rows : {len(dataset)}")

    train_ids, temp_ids = train_test_split(
        unique_ids,
        train_size=train_size,
        random_state=seed,
    )

    relative_val_size = val_size / (val_size + test_size)
    test_ids, val_ids = train_test_split(
        temp_ids,
        train_size=relative_val_size,
        random_state=seed,
    )

    print(f"[ID Split] Train IDs : {len(train_ids)}")
    print(f"[ID Split] Val IDs   : {len(val_ids)}")
    print(f"[ID Split] Test IDs  : {len(test_ids)}")

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    test_id_set = set(test_ids)

    train_dataset = dataset.filter(lambda x: x[id_column] in train_id_set, num_proc=4)
    val_dataset = dataset.filter(lambda x: x[id_column] in val_id_set, num_proc=4)
    test_dataset = dataset.filter(lambda x: x[id_column] in test_id_set, num_proc=4)

    print(f"[ID Split] Train rows : {len(train_dataset)}")
    print(f"[ID Split] Val rows   : {len(val_dataset)}")
    print(f"[ID Split] Test rows  : {len(test_dataset)}")

    torch_cols = ["input_ids", "attention_mask", "labels"]

    train_dataset.set_format(type="torch", columns=torch_cols)
    val_dataset.set_format(type="torch", columns=torch_cols)
    test_dataset.set_format(type="torch", columns=torch_cols)

    return DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )


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
    id_column: str | None,
    percentage=1.0,
):
    """
    Training and testing method
    """

    dataset_dir = f"{OUTPUT_DIR}/{DATASET_DIR}"
    model_output_dir = (
        f"{OUTPUT_DIR}/{MODEL_DIR}" if not model_output_dir else model_output_dir
    )

    assert abs(train_split + test_split + val_split - 1) < 1e-9, "Check your splits"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = f"{model_output_dir}/runs/experiment{timestamp}"

    ensure_dir(model_output_dir)
    ensure_dir(tensorboard_log_dir)
    print("Output directories created:")
    print(f"  - Model: {model_output_dir}")
    print(f"  - Logs: {tensorboard_log_dir}")

    dataset = Dataset.load_from_disk(dataset_dir)
    if percentage < 1.0:
        num_samples = int(len(dataset) * percentage)
        dataset =  dataset.shuffle(seed=42).select(range(num_samples))
    dataset = dataset.cast_column("labels", Sequence(Value("float32")))
    dataset = dataset.map(lambda x: {"length": len(x["input_ids"])}, num_proc=8)
    dataset = dataset.filter(
        filter_tokenized, batched=True, num_proc=4, batch_size=10000
    )

    if id_column:
        if id_column not in dataset.column_names:
            raise ValueError(
                f"id_column '{id_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
    dataset_dict = split_dataset_by_id(
        dataset,
        id_column=id_column,
        train_size=train_split,
        val_size=val_split,
        test_size=test_split,
        seed=42,
    )
    num_labels = len(dataset_dict["train"]["labels"][0])
    del dataset

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        attn_implementation="flash_attention_2",
        # dtype=torch.bfloat16
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
        report_to="wandb",
        logging_steps=50,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        bf16=use_bf16,
        gradient_accumulation_steps=4,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        optim="adamw_torch_fused",
        warmup_steps=0.1,
        tf32=use_bf16,
        run_name=f"run_{timestamp}",
        train_sampling_strategy="group_by_length",
        length_column_name="length",
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
    final_model_path = f"{model_output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}")

    return test_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MCI prediction model")
    parser.add_argument("--train_split", type=float, default=0.70)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--val_split", type=float, default=0.25)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_output_dir", type=str, default=None)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument(
        "--id_column",
        type=str,
        default="pat_owner_id",
        help="Dataset column to use for leak-free ID-based splitting. "
        "Omit to fall back to the original row-based split.",
    )
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
        id_column=args.id_column,
        percentage=1.0,
    )

    if results is not None:
        print(results)
