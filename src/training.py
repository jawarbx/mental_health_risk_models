"""Dataset creation script"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data_pipeline import DataPipeline

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # pylint: disable=E0602


def is_main_process():
    """Check if this is the main process"""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def wait_for_everyone():
    """Synchronize all processes - wait for main process to complete"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def label_fn_mci(
    sample: dict,
    feature_histories: dict[str, list[dict]],
    timedeltas: dict[str, str],
    feature_to_regex: dict[str, str],
):
    """Helper method to label samples based on time bounds"""
    start_time = sample["start_timestamp"]
    original_end_time = sample["end_timestamp"]
    bounds_by_delta = {
        delta_name: {
            "start_time": start_time,
            "end_time": original_end_time + delta,
        }
        for delta_name, delta in timedeltas.items()
    }

    labels = {}
    for features, history in feature_histories.items():
        regex, timestamp, data = feature_to_regex[features]
        for delta_name, bounds in bounds_by_delta.items():
            filtered_history = [
                d
                for d in history
                if bounds["end_time"]
                >= datetime.strptime(d[timestamp], "%Y-%m-%d")
                >= bounds["start_time"]
            ]
            label_hits = [d for d in filtered_history if re.search(regex, d[data])]
            if delta_name not in labels:
                labels[delta_name] = []
            labels[delta_name].append(any(label_hits))
    final_labels = {delta_name: any(checks) for delta_name, checks in labels.items()}
    label_vector = [int(final_labels[name]) for name in sorted(final_labels.keys())]
    return label_vector


def preprocess_fn_mci(
    df,
    id_feature,
    samples: dict,
    timedeltas: dict[str, "timedelta"],
    feature_to_regex: dict[str, str],
):
    """Batched method for labeling dataset and encoding"""
    batch_ids = list(set(samples[id_feature]))
    feature_histories = get_feature_histories(
        df, batch_ids, id_feature, list(feature_to_regex.keys())
    )
    label_vectors = []
    for i in range(len(samples[id_feature])):
        sample = {
            "start_timestamp": samples["start_timestamp"][i],
            "end_timestamp": samples["end_timestamp"][i],
        }
        pat_id = samples[id_feature][i]
        out = label_fn_mci(
            sample=sample,
            feature_histories=feature_histories[pat_id],
            timedeltas=timedeltas,
            feature_to_regex=feature_to_regex,
        )
        label_vectors.append(out)
    tokenized = tokenizer(
        samples["content"],
        truncation=False,
        padding=False,
    )

    tokenized["labels"] = label_vectors

    return tokenized


def __get_feature_history(df, ids: list[str], id_feature: str, feature: str):
    """Helper to get feature history for some ids from data pipeline"""
    feature_history = df[df[id_feature].isin(ids)].set_index(id_feature)
    return feature_history[feature].to_dict()


def get_feature_histories(df, ids: list[str], id_feature: str, feature_list: list[str]):
    """Method to get feature history for some ids for multiple features"""
    return {
        feature: __get_feature_history(df, ids, id_feature, feature)
        for feature in feature_list
    }


def ensure_dir(directory):
    """Ensure directory exists with error handling"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")
    except PermissionError:
        print(f"Permission denied: {directory}")
        raise
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        raise
    return directory


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
    data_output_dir=None,
    model_output_dir=None,
    per_device_batch_size=16,
    num_epochs=3,
    use_bf16=True,
    learning_rate=2e-5,
):
    """
    Main method to create dataset for training and testing
    month_delta labels are applied in ascending order
    regardless of input
    """
    main_process = is_main_process()
    dataset = None
    if data_output_dir is None:
        data_output_dir = f"{OUTPUT_DIR}/MCI_dataset_test_run"  # pylint: disable=E0602
    if model_output_dir is None:
        model_output_dir = (
            f"{OUTPUT_DIR}/MCI_model_outputs_test_run"  # pylint: disable=E0602
        )
    tensorboard_log_dir = None

    if main_process:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_log_dir = f"{model_output_dir}/runs/experiment_{timestamp}"
        ensure_dir(data_output_dir)
        ensure_dir(model_output_dir)
        ensure_dir(tensorboard_log_dir)

        print("Output directories created:")
        print(f"  - Dataset: {data_output_dir}")
        print(f"  - Model: {model_output_dir}")
        print(f"  - TensorBoard logs: {tensorboard_log_dir}")
        assert train_split + test_split + val_split == 1, "Please check your splits"

        pipeline = DataPipeline()
        samples = None
        if matching_method == "PSM":
            samples = pipeline.create_psm_samples()
        if not matching_method:
            samples = pipeline.create_regular_samples()

        month_label_to_deltas = {}
        for delta in sorted(month_deltas):
            month_label = f"{delta}_months"
            month_label_to_deltas[month_label] = relativedelta(months=delta)
        dataset = Dataset.from_list(samples)
        feature_to_regex = {
            "icd_dicts": (MCI_ICD_REGEX, 'timestamp', 'icd')  # pylint: disable=E0602
            "med_dicts": (MCI_MED_REGEX, 'timestamp', 'med') # pylint: disable=E0602
            "qa_dicts" : (MCI_QA_MEDKEY_REGEX), 'timestamp', 'med_id')# pylint: disable=E0602
        }
        dataset = dataset.map(
            lambda batch: preprocess_fn_mci(
                pipeline.all_data,
                "pat_owner_id",
                batch,
                month_label_to_deltas,
                feature_to_regex,
            ),
            batched=True,
        )
        dataset.save_to_disk(data_output_dir)
        print(f"Labeled dataset saved to {data_output_dir}")
        del pipeline
    wait_for_everyone()

    if dist.is_available() and dist.is_initialized():
        if main_process:
            tensorboard_log_dir_list = [tensorboard_log_dir]
        else:
            tensorboard_log_dir_list = [None]

        dist.broadcast_object_list(tensorboard_log_dir_list, src=0)
        tensorboard_log_dir = tensorboard_log_dir_list[0]
    else:
        if tensorboard_log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tensorboard_log_dir = f"{model_output_dir}/runs/experiment_{timestamp}"

    if not main_process:
        print(
            f"Process {os.environ.get('LOCAL_RANK')}: Using TensorBoard dir: {tensorboard_log_dir}"
        )

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

    if main_process:
        print(f"Train size: {len(dataset_dict['train'])}")
        print(f"Validation size: {len(dataset_dict['validation'])}")
        print(f"Test size: {len(dataset_dict['test'])}")

    # Free memory from data pipeline
    del dataset, train_temp, train_val

    num_labels = len(month_deltas)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels  # pylint: disable=E0602
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

    if main_process:
        print(
            f"Effective batch size: {per_device_batch_size * torch.cuda.device_count()}"
        )

    trainer.train()
    if main_process:
        print("Evaluating on test set...")
        test_results = trainer.evaluate(dataset_dict["test"])
        print(f"Test results: {test_results}")
    else:
        test_results = None

    if trainer.is_world_process_zero():
        final_model_path = f"{model_output_dir}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Final model saved to {final_model_path}")

    return dataset_dict, trainer, test_results if main_process else None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MCI prediction model")
    parser.add_argument(
        "--month_deltas",
        nargs="+",
        type=int,
        default=[6, 12, 18],
        help="Month deltas for prediction windows",
    )
    parser.add_argument(
        "--matching_method",
        type=str,
        default=None,
        choices=["PSM", None],
        help="Matching method for samples",
    )
    parser.add_argument("--train_split", type=float, default=0.70)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--val_split", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--data_output_dir", type=str, default=None)
    parser.add_argument("--model_output_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = main(
        matching_method=args.matching_method,
        month_deltas=args.month_deltas,
        train_split=args.train_split,
        test_split=args.test_split,
        val_split=args.val_split,
        data_output_dir=args.data_output_dir,
        model_output_dir=args.model_output_dir,
        per_device_batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )

    if results and results[2] is not None:
        print(results[2])
