"""Threshold Selection via ROC Curve for Multi-Label Classification"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, Sequence, Value
from dotenv import load_dotenv
from scipy.special import expit
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from preprocess import ensure_dir
from training import filter_tokenized, split_dataset_by_id

# ── Environment ────────────────────────────────────────────────────
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

LABEL_NAMES = [
    "Dx within 18mo",
    "Dx within 24mo",
    "Dx within 36mo",
]


# ── Model Loading ──────────────────────────────────────────────────
def load_model_with_lora(model_path: str, num_labels: int):
    """Load LoRA adapter model"""
    from peft import PeftConfig, PeftModel

    peft_config     = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    print(f"Base model : {base_model_name}")
    print(f"Adapter    : {model_path}")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    print("LoRA adapter loaded ✅")
    return model


# ── ROC Threshold Selection ────────────────────────────────────────
def find_optimal_thresholds_roc(
    logits:     np.ndarray,
    labels:     np.ndarray,
    output_dir: str,
) -> dict:
    """
    For each label:
      1. Compute ROC curve
      2. Find optimal threshold via Youden's J
         J = TPR - FPR
      3. Plot and save ROC curves

    Returns:
        dict of {label_name: optimal_threshold}
    """
    probs    = expit(logits)
    n_labels = labels.shape[1]

    ensure_dir(output_dir)

    fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 6))
    if n_labels == 1:
        axes = [axes]

    fig.suptitle("ROC Curves — Optimal Threshold via Youden's J", fontsize=14)

    optimal_thresholds = {}

    for i in range(n_labels):
        ax         = axes[i]
        label_name = LABEL_NAMES[i]

        # ── ROC curve ─────────────────────────────────────────────
        fpr, tpr, thresholds = roc_curve(labels[:, i], probs[:, i])
        roc_auc_val          = auc(fpr, tpr)

        # ── Youden's J — optimal threshold ────────────────────────
        j_scores          = tpr - fpr
        optimal_ix        = np.argmax(j_scores)
        optimal_threshold = float(thresholds[optimal_ix])
        optimal_fpr       = fpr[optimal_ix]
        optimal_tpr       = tpr[optimal_ix]
        optimal_j         = j_scores[optimal_ix]

        optimal_thresholds[label_name] = optimal_threshold

        print(f"\n── Label {i}: {label_name} ──────────────────────────")
        print(f"  AUC                  : {roc_auc_val:.4f}")
        print(f"  Optimal threshold    : {optimal_threshold:.4f}")
        print(f"  Youden's J           : {optimal_j:.4f}")
        print(f"  Sensitivity (Recall) : {optimal_tpr:.4f}")
        print(f"  Specificity          : {1 - optimal_fpr:.4f}")
        print(f"  FPR                  : {optimal_fpr:.4f}")

        # ── Plot ───────────────────────────────────────────────────
        ax.plot(fpr, tpr,
                color="steelblue", lw=2,
                label=f"AUC = {roc_auc_val:.3f}")

        ax.plot([0, 1], [0, 1],
                color="gray", linestyle="--",
                alpha=0.5, label="Random")

        ax.scatter(optimal_fpr, optimal_tpr,
                   color="red", s=120, zorder=5,
                   label=f"Optimal = {optimal_threshold:.3f}")

        ax.axvline(optimal_fpr, color="red", linestyle=":", alpha=0.4)
        ax.axhline(optimal_tpr, color="red", linestyle=":", alpha=0.4)

        ax.annotate(
            f"  threshold={optimal_threshold:.3f}\n"
            f"  TPR={optimal_tpr:.3f}\n"
            f"  FPR={optimal_fpr:.3f}",
            xy=(optimal_fpr, optimal_tpr),
            xytext=(optimal_fpr + 0.05, optimal_tpr - 0.15),
            fontsize=9,
            color="red",
        )

        ax.set_title(f"Label {i}: {label_name}")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = f"{output_dir}/roc_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nROC curves saved → {plot_path}")

    return optimal_thresholds


# ── Evaluation ─────────────────────────────────────────────────────
def evaluate_at_optimal_thresholds(
    logits:             np.ndarray,
    labels:             np.ndarray,
    optimal_thresholds: dict,
    split:              str = "test",
) -> dict:
    """Apply per-label optimal thresholds and compute all metrics"""
    probs = expit(logits)
    preds = np.zeros_like(labels, dtype=int)

    print(f"\n── Applying Optimal Thresholds ({split}) ─────────────")
    for i, (label_name, threshold) in enumerate(optimal_thresholds.items()):
        preds[:, i] = (probs[:, i] >= threshold).astype(int)
        print(f"  {label_name}: threshold = {threshold:.4f}")

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

    cls_report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        zero_division=0,
    )

    print(f"\n── {split} Results ────────────────────────────────────")
    print(f"  F1 micro        : {f1_micro:.4f}")
    print(f"  F1 macro        : {f1_macro:.4f}")
    print(f"  F1 weighted     : {f1_weighted:.4f}")
    print(f"  Precision macro : {precision_macro:.4f}")
    print(f"  Recall macro    : {recall_macro:.4f}")
    print(f"  ROC-AUC         : {roc_auc_weighted:.4f}")
    print(f"\nClassification Report:\n{cls_report}")

    return {
        "split":                 split,
        "precision_macro":       precision_macro,
        "recall_macro":          recall_macro,
        "f1_macro":              f1_macro,
        "precision_micro":       precision_micro,
        "recall_micro":          recall_micro,
        "f1_micro":              f1_micro,
        "precision_weighted":    precision_weighted,
        "recall_weighted":       recall_weighted,
        "f1_weighted":           f1_weighted,
        "roc_auc_weighted":      roc_auc_weighted,
        "classification_report": cls_report,
    }


# ── Main ───────────────────────────────────────────────────────────
def run_roc_threshold_selection(
    model_path:            str,
    dataset_dir:           str,
    output_dir:            str,
    per_device_batch_size: int  = 8,
    use_bf16:              bool = True,
    use_lora:              bool = True,
):
    print(f"\n{'='*60}")
    print("  ROC Threshold Selection")
    print(f"  Model      : {model_path}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ── Load tokenizer ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ── Load dataset ───────────────────────────────────────────────
    print("Loading dataset...")
    dataset = Dataset.load_from_disk(dataset_dir)
    dataset = dataset.cast_column("labels", Sequence(Value("float32")))
    dataset = dataset.map(
        lambda x: {"length": len(x["input_ids"])},
        num_proc=8
    )
    dataset = dataset.filter(
        filter_tokenized, batched=True, num_proc=4, batch_size=10000
    )

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

    # ── Load model ─────────────────────────────────────────────────
    if use_lora:
        model = load_model_with_lora(model_path, num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            attn_implementation="flash_attention_2",
        )

    # ── Trainer for inference only ─────────────────────────────────
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
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    # ── Step 1: Validation logits ──────────────────────────────────
    print(f"Running inference on validation set "
          f"({len(dataset_dict['validation']):,} samples)...")
    val_predictions = trainer.predict(dataset_dict["validation"])
    val_logits      = np.asarray(val_predictions.predictions)
    val_labels      = np.asarray(val_predictions.label_ids)
    print(f"Logits shape : {val_logits.shape}")
    print(f"Labels shape : {val_labels.shape}\n")

    # ── Step 2: Find optimal thresholds via ROC ────────────────────
    print("Computing ROC curves on validation set...")
    optimal_thresholds = find_optimal_thresholds_roc(
        val_logits,
        val_labels,
        output_dir=f"{output_dir}/roc_plots",
    )

    # ── Step 3: Evaluate validation ───────────────────────────────
    val_results = evaluate_at_optimal_thresholds(
        val_logits,
        val_labels,
        optimal_thresholds,
        split="validation",
    )

    # ── Step 4: Test logits ────────────────────────────────────────
    print(f"\nRunning inference on test set "
          f"({len(dataset_dict['test']):,} samples)...")
    test_predictions = trainer.predict(dataset_dict["test"])
    test_logits      = np.asarray(test_predictions.predictions)
    test_labels      = np.asarray(test_predictions.label_ids)
    print(f"Logits shape : {test_logits.shape}")
    print(f"Labels shape : {test_labels.shape}\n")

    # ── Step 5: Evaluate test ──────────────────────────────────────
    test_results = evaluate_at_optimal_thresholds(
        test_logits,
        test_labels,
        optimal_thresholds,
        split="test",
    )

    # ── Step 6: Save ───────────────────────────────────────────────
    ensure_dir(output_dir)

    # Optimal thresholds CSV
    threshold_df = pd.DataFrame([{
        "label":     name,
        "threshold": threshold,
    } for name, threshold in optimal_thresholds.items()])
    threshold_df.to_csv(f"{output_dir}/optimal_thresholds.csv", index=False)

    # Val/Test metrics CSV
    metrics_df = pd.DataFrame([
        {k: v for k, v in val_results.items()  if k != "classification_report"},
        {k: v for k, v in test_results.items() if k != "classification_report"},
    ])
    metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)

    # Classification reports TXT
    report_path = f"{output_dir}/classification_reports.txt"
    with open(report_path, "w") as f:
        f.write("OPTIMAL THRESHOLDS (from validation ROC)\n")
        f.write("=" * 50 + "\n")
        for name, threshold in optimal_thresholds.items():
            f.write(f"  {name}: {threshold:.4f}\n")
        f.write("\n\nVALIDATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(val_results["classification_report"])
        f.write("\n\nTEST REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(test_results["classification_report"])

    print(f"\n{'='*60}")
    print("  SAVED")
    print(f"  {output_dir}/optimal_thresholds.csv")
    print(f"  {output_dir}/metrics.csv")
    print(f"  {output_dir}/roc_plots/roc_curves.png")
    print(f"  {output_dir}/classification_reports.txt")
    print(f"{'='*60}")

    return optimal_thresholds, val_results, test_results

# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="ROC threshold selection for trained model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (defaults to final_model in MODEL_DIR)",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--use_lora",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to load model with LoRA adapter (default: True)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_path  = args.model_path or f"{OUTPUT_DIR}/{MODEL_DIR}/final_model"
    dataset_dir = f"{OUTPUT_DIR}/{DATASET_DIR}"
    output_dir  = f"{OUTPUT_DIR}/{MODEL_DIR}/roc_threshold"

    optimal_thresholds, val_results, test_results = run_roc_threshold_selection(
        model_path=model_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        per_device_batch_size=args.per_device_batch_size,
        use_bf16=args.use_bf16,
        use_lora=args.use_lora,
    )
