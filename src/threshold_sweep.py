"""Threshold Sweep Script for Multi-Label Classification"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, Sequence, Value
from dotenv import load_dotenv
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
from training import split_dataset_by_id, filter_tokenized

# ── Environment ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

MODEL_NAME  = os.getenv("MODEL_NAME")
DATASET_DIR = os.getenv("DATASET_DIR")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR")
MODEL_DIR   = os.getenv("MODEL_DIR")

if not all([MODEL_NAME, DATASET_DIR, OUTPUT_DIR, MODEL_DIR]):
    missing = [
        var
        for var, val in {
            "MODEL_NAME":   MODEL_NAME,
            "DATASET_DIR":  DATASET_DIR,
            "OUTPUT_DIR":   OUTPUT_DIR,
            "MODEL_DIR":    MODEL_DIR,
        }.items()
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# ── Thresholds to sweep ────────────────────────────────────────────────────────
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.6, 0.7, 0.9]


# ── Metric helper ──────────────────────────────────────────────────────────────
def evaluate_at_threshold(logits: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Compute all metrics for a single threshold value."""
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    try:
        roc_auc_weighted = roc_auc_score(labels, probs, average="weighted")
    except ValueError:
        roc_auc_weighted = float("nan")

    cls_report = classification_report(labels, preds, zero_division=0)

    return {
        "threshold":          threshold,
        # macro
        "precision_macro":    precision_macro,
        "recall_macro":       recall_macro,
        "f1_macro":           f1_macro,
        # micro
        "precision_micro":    precision_micro,
        "recall_micro":       recall_micro,
        "f1_micro":           f1_micro,
        # weighted
        "precision_weighted": precision_weighted,
        "recall_weighted":    recall_weighted,
        "f1_weighted":        f1_weighted,
        # auc
        "roc_auc_weighted":   roc_auc_weighted,
        # full report
        "classification_report": cls_report,
    }


def print_results(metrics: dict, title: str = "Results"):
    """Pretty print a metrics dictionary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Threshold       : {metrics['threshold']:.2f}")
    print(f"  F1 micro        : {metrics['f1_micro']:.4f}")
    print(f"  F1 macro        : {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted     : {metrics['f1_weighted']:.4f}")
    print(f"  Precision macro : {metrics['precision_macro']:.4f}")
    print(f"  Recall macro    : {metrics['recall_macro']:.4f}")
    print(f"  ROC-AUC         : {metrics['roc_auc_weighted']:.4f}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

def load_model_with_lora(model_path: str, num_labels: int):
    """Load LoRA adapter model"""
    from peft import PeftModel, PeftConfig

    # ── Read adapter config to get base model ─────────────────────
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    print(f"Base model : {base_model_name}")
    print(f"Adapter    : {model_path}")

    # ── Load base model ───────────────────────────────────────────
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        attn_implementation="flash_attention_2",
    )

    # ── Load LoRA adapter on top ──────────────────────────────────
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
    )

    print("LoRA adapter loaded")
    return model

# ── Main sweep ─────────────────────────────────────────────────────────────────
def run_threshold_sweep(
    model_path: str,
    dataset_dir: str,
    output_dir: str,
    per_device_batch_size: int = 64,
    use_bf16: bool = True,
    thresholds: list[float] = THRESHOLDS,
    use_lora=True
):
    """
    1. Sweep thresholds on validation set to find best threshold
    2. Evaluate best threshold on test set
    """
    print(f"\n{'='*60}")
    print("  Threshold Sweep")
    print(f"  Model : {model_path}")
    print(f"{'='*60}\n")

    # ── Load tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ── Load dataset ───────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = Dataset.load_from_disk(dataset_dir)
    dataset = dataset.cast_column("labels", Sequence(Value("float32")))
    dataset = dataset.map(lambda x: {"length": len(x["input_ids"])}, num_proc=8)
    dataset = dataset.filter(
        filter_tokenized, batched=True, num_proc=4, batch_size=10000
    )

    # Reuse the same ID-based split from training script
    dataset_dict = split_dataset_by_id(
        dataset,
        id_column="pat_owner_id",
        train_size=0.70,
        val_size=0.05,
        test_size=0.25,
        seed=42,
    )
    num_labels = len(dataset_dict["train"]["labels"][0])
    del dataset

    # ── Load Model ─────────────────────────────────────────────────
    if use_lora:
        model = load_model_with_lora(model_path, num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                problem_type="multi_label_classification",
                attn_implementation="flash_attention_2",
            )

    # ── Minimal TrainingArguments just for prediction ──────────────────────────
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_batch_size,
        bf16=use_bf16,
        tf32=use_bf16,
        dataloader_num_workers=24,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=16,
        dataloader_persistent_workers=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    # ── Step 1: Collect validation logits ─────────────────────────────────────
    print(f"Running inference on validation set ({len(dataset_dict['validation']):,} samples)...")
    val_predictions = trainer.predict(dataset_dict["validation"])
    val_logits      = np.asarray(val_predictions.predictions)
    val_labels      = np.asarray(val_predictions.label_ids)
    print(f"Logits shape : {val_logits.shape}")
    print(f"Labels shape : {val_labels.shape}")

    # ── Save validation logits ─────────────────────────────────────────
    ensure_dir(output_dir)
    np.save(f"{output_dir}/val_logits.npy", val_logits)
    np.save(f"{output_dir}/val_labels.npy", val_labels)
    print(f"Validation logits saved → {output_dir}/val_logits.npy")
    print(f"Validation labels saved → {output_dir}/val_labels.npy\n")

    # ── Step 2: Sweep thresholds on validation ─────────────────────────────────
    print("Sweeping thresholds on validation set...")
    print(f"{'─'*80}")
    val_results = []
    for threshold in thresholds:
        metrics = evaluate_at_threshold(val_logits, val_labels, threshold)
        val_results.append(metrics)

        print(f"Threshold {threshold:.2f} │ "
              f"F1 micro={metrics['f1_micro']:.4f} │ "
              f"F1 macro={metrics['f1_macro']:.4f} │ "
              f"Recall macro={metrics['recall_macro']:.4f} │ "
              f"Precision macro={metrics['precision_macro']:.4f}")

    # ── Step 3: Validation summary table ──────────────────────────────────────
    summary_cols = [
        "threshold",
        "f1_micro", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro",
        "roc_auc_weighted",
    ]
    df_val = pd.DataFrame(val_results)[summary_cols]

    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY TABLE")
    print(f"{'='*60}")
    print(df_val.to_string(index=False, float_format="{:.4f}".format))

    # ── Step 4: Find best threshold ────────────────────────────────────────────
    best_val = max(val_results, key=lambda x: x["f1_micro"])
    best_threshold = best_val["threshold"]

    print_results(best_val, title=f"BEST THRESHOLD ON VALIDATION: {best_threshold:.2f}")

    # ── Step 5: Collect test logits ────────────────────────────────────────────
    print(f"Running inference on test set ({len(dataset_dict['test']):,} samples)...")
    test_predictions = trainer.predict(dataset_dict["test"])
    test_logits      = np.asarray(test_predictions.predictions)
    test_labels      = np.asarray(test_predictions.label_ids)
    print(f"Logits shape : {test_logits.shape}")
    print(f"Labels shape : {test_labels.shape}")

    # ── Save test logits ───────────────────────────────────────────────
    np.save(f"{output_dir}/test_logits.npy", test_logits)
    np.save(f"{output_dir}/test_labels.npy", test_labels)
    print(f"Test logits saved → {output_dir}/test_logits.npy")
    print(f"Test labels saved → {output_dir}/test_labels.npy\n")

    # ── Step 6: Evaluate test set at best threshold ────────────────────────────
    # Method 1 — sweep best threshold
    test_sweep = evaluate_at_threshold(test_logits, test_labels, best_threshold)
    print_results(test_sweep, title=f"TEST RESULTS — Sweep threshold={best_threshold:.2f}")

    # ── Step 7: Save all results ───────────────────────────────────────────────
    ensure_dir(output_dir)

    # Validation sweep CSV
    val_csv_path = f"{output_dir}/validation_threshold_sweep.csv"
    df_val.to_csv(val_csv_path, index=False)
    print(f"Validation sweep saved to  : {val_csv_path}")

    # Test results CSV
    test_summary = {k: v for k, v in test_results.items() if k != "classification_report"}
    df_test = pd.DataFrame([test_summary])
    test_csv_path = f"{output_dir}/test_results.csv"
    df_test.to_csv(test_csv_path, index=False)
    print(f"Test results saved to      : {test_csv_path}")

    # Classification report txt
    report_path = f"{output_dir}/test_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best Threshold (from validation): {best_threshold:.2f}\n\n")
        f.write(test_results["classification_report"])
    print(f"Classification report saved: {report_path}")

    return val_results, best_val, test_results


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Threshold sweep for trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (defaults to final_model in MODEL_DIR)",
    )
    parser.add_argument("--per_device_batch_size", type=int, default=32)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=THRESHOLDS,
        help="List of thresholds to sweep e.g. --thresholds 0.1 0.2 0.3",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_path  = args.model_path or f"{OUTPUT_DIR}/{MODEL_DIR}/final_model"
    dataset_dir = f"{OUTPUT_DIR}/{DATASET_DIR}"
    output_dir  = f"{OUTPUT_DIR}/{MODEL_DIR}/threshold_sweep"

    val_results, best_val, test_results = run_threshold_sweep(
        model_path=model_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        per_device_batch_size=args.per_device_batch_size,
        use_bf16=args.use_bf16,
        thresholds=args.thresholds,
    )
