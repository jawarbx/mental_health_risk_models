"""Dataset Relabeling Method"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from tqdm import tqdm

# from transformers import AutoTokenizer

from preprocess import gap_filter_batched, label_fn_mci, ensure_dir, parse_timestamp

from data_pipeline import DataPipeline

SCRIPT_DIR = Path(__file__).parent.resolve()
dotenv_path = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)
tqdm.pandas()

MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DATASET_DIR = os.getenv("DATASET_DIR")
RELABELED_DIR = os.getenv("RELABELED_DIR")
MCI_QA_MEDKEY_PATH = os.getenv("MCI_QA_MEDKEY_PATH")
MCI_ICD_REGEX = os.getenv("MCI_ICD_REGEX")
MCI_MED_REGEX = os.getenv("MCI_MED_REGEX")

if not all([MODEL_NAME, OUTPUT_DIR, MCI_QA_MEDKEY_PATH, MCI_ICD_REGEX, MCI_MED_REGEX]):
    missing = [
        var
        for var, val in {
            "MODEL_NAME": MODEL_NAME,
            "OUTPUT_DIR": OUTPUT_DIR,
            "DATASET_DIR": DATASET_DIR,
            "MCI_QA_MEDKEY_PATH": MCI_QA_MEDKEY_PATH,
            "MCI_ICD_REGEX": MCI_ICD_REGEX,
            "MCI_MED_REGEX": MCI_MED_REGEX,
        }.items()
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def relabel_fn_mci(
    df,
    id_feature,
    samples: dict,
    timedeltas: dict[str, str],
    feature_to_regex: dict[str, str],
    gap=0,
):
    label_vectors = [
        label_fn_mci(
            sample={"start_timestamp": start_ts, "end_timestamp": end_ts},
            feature_map=df.get(pat_id, {}),
            timedeltas=timedeltas,
            feature_to_regex=feature_to_regex,
            gap=gap,
        )
        for pat_id, start_ts, end_ts in zip(
            samples[id_feature],
            samples["start_timestamp"],
            samples["end_timestamp"],
        )
    ]
    return {"labels": label_vectors}


def main(
    month_deltas: list[int],
    relabeled_dataset_dir: str = None,
    dataset_dir=None,
    gap=0,
):
    """
    Main method to relabel dataset for training and testing
    month_delta labels are applied in ascending order
    regardless of input
    """
    if not dataset_dir:
        dataset_dir = f"{OUTPUT_DIR}/{DATASET_DIR}"

    if not relabeled_dataset_dir:
        relabeled_dataset_dir = f"{OUTPUT_DIR}/{RELABELED_DIR}"
    ensure_dir(relabeled_dataset_dir)

    assert os.path.isdir(dataset_dir), "Please check if dataset exists"
    assert os.path.isdir(relabeled_dataset_dir), "Please choose a path for destination"

    pipeline = DataPipeline()
    df = pipeline.all_data
    del pipeline
    dataset = Dataset.load_from_disk(dataset_dir)
    month_label_to_deltas = {}
    for delta in sorted(month_deltas):
        month_label = f"{delta}_months"
        month_label_to_deltas[month_label] = relativedelta(months=delta)
    month_gap = relativedelta(months=gap) if gap > 0 else 0
    if month_gap:
        dataset = dataset.filter(
            lambda samples: gap_filter_batched(
                samples, month_gap, month_label_to_deltas
            ),
            batched=True,
            batch_size=10000,
        )
    print("Loading key")
    with open(
        MCI_QA_MEDKEY_PATH,
        "r",
        encoding="utf-8",  # pylint: disable=E0602
    ) as json_file:
        key_ids = json.load(json_file)
    key_ids = set(sum(key_ids.values(), []))
    feature_to_regex = {
        "icd_dicts": (MCI_ICD_REGEX, "timestamp", "icd"),  # pylint: disable=E0602
        "med_history": (
            MCI_MED_REGEX,  # pylint: disable=E0602
            "timestamp",
            "med_list",
        ),
        "med_data": (
            key_ids,
            "timestamp",
            "med_id",
        ),
    }
    for feature in feature_to_regex.keys():
        df[feature] = df[feature].progress_apply(
            lambda h: (
                [
                    {**d, "timestamp": parse_timestamp(d["timestamp"])}
                    for d in h
                    if d is not None
                ]
                if isinstance(h, list)
                else h
            )
        )
    df_grouped = df.set_index("pat_owner_id")[list(feature_to_regex.keys())].to_dict(
        orient="index"
    )
    dataset = dataset.map(
        lambda batch: relabel_fn_mci(
            df_grouped,
            "pat_owner_id",
            batch,
            month_label_to_deltas,
            feature_to_regex,
            gap=month_gap,
        ),
        batched=True,
        batch_size=10000,
        writer_batch_size=50000,
        keep_in_memory=True,
        desc="Relabeling MCI",
    )
    dataset.save_to_disk(relabeled_dataset_dir)
    print(f"Labeled dataset saved to {relabeled_dataset_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Relabel dataset (MCI or Depression)")
    parser.add_argument(
        "--relabeled_dataset_dir",
        type=str,
        help="Destination of relabeled dataset",
        default=None,
    )

    parser.add_argument(
        "--month_deltas",
        nargs="+",
        type=int,
        default=[18, 24, 36],
        help="Month deltas for prediction windows",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=12,
        help="History / Qualifier gap in months",
    )

    parser.add_argument("--dataset_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        month_deltas=args.month_deltas,
        dataset_dir=args.dataset_dir,
        gap=int(args.gap),
        relabeled_dataset_dir=args.relabeled_dataset_dir,
    )
