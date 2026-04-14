"""Model Training script"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset, DatasetDict, Value, concatenate_datasets
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
)

from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback
)

from preprocess import ensure_dir
from focal_trainer import FocalLoss, BinaryFocalLossTrainer

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
    preds = (probs >= 0.05).astype(int)

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

def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred

    if logits.ndim > 1:
        logits = logits.squeeze(-1)

    probs = 1 / (1 + np.exp(-logits))

    # Find optimal threshold by F1 on validation set
    thresholds = np.arange(0.01, 0.50, 0.01)
    f1_scores = [
            f1_score(labels, (probs >= t).astype(int), zero_division=0)
            for t in thresholds
            ]
    best_thresh = thresholds[np.argmax(f1_scores)]
    preds_best = (probs >= best_thresh).astype(int)
    preds_05 = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds_05, labels=[0, 1]).ravel()
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(labels, preds_best, labels=[0, 1]).ravel()

    return {
    "auroc": roc_auc_score(labels, probs),
    "auprc": average_precision_score(labels, probs),
    "f1": f1_score(labels, preds_05, zero_division=0),
    "precision": precision_score(labels, preds_05, zero_division=0),
    "recall": recall_score(labels, preds_05, zero_division=0),
    "tp": int(tp),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "best_threshold": float(best_thresh),
    "f1_best": float(max(f1_scores)),
    "precision_best": precision_score(labels, preds_best, zero_division=0),
    "recall_best": recall_score(labels, preds_best, zero_division=0),
    "mcc_best": matthews_corrcoef(labels, preds_best),
    "tp_best": int(tp_b),
    "tn_best": int(tn_b),
    "fp_best": int(fp_b),
    "fn_best": int(fn_b),
    "mean_prob": float(probs.mean()),
    "mean_prob_pos": float(probs[labels == 1].mean()),
    "mean_prob_neg": float(probs[labels == 0].mean()),
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

def undersample_negatives(
    dataset:   Dataset,
    label_col: str = "labels",
    neg_ratio: int = 4,
    seed:      int = 42,
) -> Dataset:
    """
    Undersample negative samples (all-zero labels) in training set.
    Keeps all positive samples.
    Keeps neg_ratio negatives per positive sample.
    """
    labels = np.array(dataset[label_col])

    # ── Split indices ──────────────────────────────────────────────
    pos_idx = np.where(labels > 0)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos      = len(pos_idx)
    n_neg_keep = min(n_pos * neg_ratio, len(neg_idx))

    print("\n── Undersampling Training Set ────────────────────────")
    print(f"  Positives        : {n_pos:,}  ({n_pos/len(dataset)*100:.2f}%)")
    print(f"  Negatives (orig) : {len(neg_idx):,}  ({len(neg_idx)/len(dataset)*100:.2f}%)")
    print(f"  Negatives (keep) : {n_neg_keep:,}  (ratio 1:{neg_ratio})")
    print(f"  Total (before)   : {len(dataset):,}")

    # ── Sample negatives ───────────────────────────────────────────
    rng         = np.random.default_rng(seed)
    neg_sampled = rng.choice(neg_idx, size=n_neg_keep, replace=False)

    # ── Combine & shuffle ──────────────────────────────────────────
    keep_idx = np.concatenate([pos_idx, neg_sampled])
    rng.shuffle(keep_idx)

    balanced = dataset.select(keep_idx.tolist())

    print(f"  Total (after)    : {len(balanced):,}")
    print(f"  New positive rate: {n_pos/len(balanced)*100:.1f}%")
    print("──────────────────────────────────────────────────────")

    return balanced

def split_with_train_only(
    ds: Dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.05,
    test_ratio: float = 0.25,
    train_only_col: str = "_train_only",
    seed: int = 42,
) -> DatasetDict:
    """
    Split a Hugging Face Dataset into train/validation/test with given ratios,
    enforcing that all rows where `train_only_col` is True go to train.

    Ratios are global (over the entire dataset) and will be respected as closely
    as possible given the constraint.
    """

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Separate mandatory-train and flexible samples
    ds_train_only = ds.filter(lambda x: x[train_only_col])
    ds_flexible   = ds.filter(lambda x: not x[train_only_col])

    n_total      = len(ds)
    n_train_only = len(ds_train_only)
    n_flex       = len(ds_flexible)

    # Desired global counts
    n_train_target = int(round(train_ratio * n_total))
    n_val_target   = int(round(val_ratio   * n_total))

    # Flexible samples needed for each split (cannot be negative)
    n_flex_train = max(0, n_train_target - n_train_only)
    # Can't exceed available flexible samples
    n_flex_train = min(n_flex_train, n_flex)

    # Put as many as possible into val up to the target
    remaining_after_train = n_flex - n_flex_train
    n_flex_val = min(n_val_target, remaining_after_train)

    # Rest go to test
    n_flex_test = n_flex - n_flex_train - n_flex_val

    # Optional: sanity checks
    assert n_flex_train + n_flex_val + n_flex_test == n_flex

    # Randomly assign flexible samples to splits
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(n_flex)
    rng.shuffle(indices)

    train_idx = indices[:n_flex_train]
    val_idx   = indices[n_flex_train:n_flex_train + n_flex_val]
    test_idx  = indices[n_flex_train + n_flex_val:]

    flex_train = ds_flexible.select(train_idx)
    flex_val   = ds_flexible.select(val_idx)
    flex_test  = ds_flexible.select(test_idx)

    # Final splits
    train = concatenate_datasets([ds_train_only, flex_train])
    val   = flex_val
    test  = flex_test

    train_counts = Counter(train['labels'])
    val_counts = Counter(val['labels'])
    test_counts = Counter(test['labels'])

    print(f"[Split] Train rows : {len(train)}")
    print(f"[Split] Val rows   : {len(val)}")
    print(f"[Split] Test rows  : {len(test)}")

    print(f"[Label Counts] Train labels: {train_counts}")
    print(f"[Label Counts] Val labels: {val_counts}")
    print(f"[Label Counts] Test labels: {test_counts}")


    return DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })

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
    #Rough patch for age
    demos = pd.read_csv('/gpfs/data/mankowskilab/NYU_Consult_raw/demographics_revised.csv')
    fifty_plus = set(demos[demos.age >= 50].pat_owner_id)
    dataset = dataset.filter(lambda example: example["pat_owner_id"] in fifty_plus)

    if percentage < 1.0:
        num_samples = int(len(dataset) * percentage)
        dataset =  dataset.shuffle(seed=42).select(range(num_samples))
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.cast_column("labels", (Value("int64")))
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
    dataset_dict = split_with_train_only(
        dataset,
    )

    dataset_dict = DatasetDict({
        "train":       dataset_dict["train"],
        "validation":   dataset_dict["validation"],
        "test":         dataset_dict["test"],
    })

    num_labels = 1 #len(dataset_dict["train"]["labels"][0])
    del dataset

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        attn_implementation="flash_attention_2",
    )

    if use_lora:
        print("Using lora")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # sequence classification
            r=16,
            lora_alpha=32,
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
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        logging_dir=tensorboard_log_dir,
        logging_first_step=True,
        report_to="wandb",
        logging_steps=50,
        save_total_limit=8,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        bf16=use_bf16,
        gradient_accumulation_steps=4,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        optim="adamw_torch_fused",
        warmup_steps=200,
        tf32=use_bf16,
        run_name=f"run_{timestamp}",
        train_sampling_strategy="group_by_length",
        length_column_name="length",
    )
    #pos_weight = torch.tensor(np.sqrt([262366 / 12240]))

    loss_fn = FocalLoss(
            gamma=2.0,
            alpha=0.75,              # weight of positive class
            reduction="sum",
            task_type="binary",
            pos_weight=None #torch.tensor([2.0])
    )

    trainer = BinaryFocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_binary,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        loss_fn=loss_fn
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
    parser.add_argument("--test_split", type=float, default=0.25)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--per_device_batch_size", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
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
