"""Dataset creation script"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from transformers import AutoTokenizer

from data_pipeline import DataPipeline

SCRIPT_DIR = Path(__file__).parent.resolve()
dotenv_path = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DATASET_DIR = os.getenv("DATASET_DIR")
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def label_fn_mci(
    sample: dict,
    feature_map: dict[str, list[dict]],
    timedeltas: dict[str, str],
    feature_to_regex: dict[str, str],
    gap,
):
    """Helper method to label samples based on time bounds"""
    original_end_time = parse_timestamp(sample["end_timestamp"])
    bounds_by_delta = {
        delta_name: {
            "start_time": original_end_time + gap,
            "end_time": original_end_time + delta,
        }
        for delta_name, delta in timedeltas.items()
    }
    labels = {}
    for feature, history in feature_map.items():
        regex, timestamp, data = feature_to_regex[feature]
        for delta_name, bounds in bounds_by_delta.items():
            if (
                history is None
                or (isinstance(history, float) and math.isnan(history))
                or (hasattr(history, "__len__") and len(history) == 0)
            ):
                filtered_history = []
                label_hits = []
            else:
                filtered_history = [
                    d
                    for d in history
                    if d is not None
                    and bounds["end_time"]
                    >= parse_timestamp(d[timestamp])
                    >= bounds["start_time"]
                ]
                label_hits = [
                    d for d in filtered_history if parse_qualifier(regex, d[data])
                ]
            if delta_name not in labels:
                labels[delta_name] = []
            labels[delta_name].append(any(label_hits))
    final_labels = {delta_name: any(checks) for delta_name, checks in labels.items()}
    label_vector = [int(final_labels[name]) for name in sorted(final_labels.keys())]
    print(final_labels.keys())
    return label_vector


def preprocess_fn_mci(
    df,
    id_feature,
    samples: dict,
    timedeltas: dict[str, str],
    feature_to_regex: dict[str, str],
    gap=0,
):
    """Batched method for labeling dataset and encoding"""
    batch_ids = list(set(samples[id_feature]))
    features = list(feature_to_regex.keys())
    feature_histories = get_feature_histories(df, batch_ids, id_feature, features)
    id_to_feature = defaultdict(dict)
    for feature, id_map in feature_histories.items():
        for pat_id, data in id_map.items():
            id_to_feature[pat_id][feature] = data
    id_to_feature = dict(id_to_feature)
    label_vectors = []
    for i in range(len(samples[id_feature])):
        sample = {
            "start_timestamp": samples["start_timestamp"][i],
            "end_timestamp": samples["end_timestamp"][i],
        }
        pat_id = samples[id_feature][i]
        out = label_fn_mci(
            sample=sample,
            feature_map=id_to_feature[pat_id],
            timedeltas=timedeltas,
            feature_to_regex=feature_to_regex,
            gap=gap,
        )
        label_vectors.append(out)
    tokenized = tokenizer(
        samples['content'],
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


def parse_timestamp(ts_string):
    """Helper to parse timestamps across multiple precisions"""
    if isinstance(ts_string, int):
        try:
            ts_string = ts_string / 1000 if ts_string > 1e12 else ts_string
            return datetime.fromtimestamp(ts_string)
        except Exception:
            raise Exception(f"Time data '{ts_string}' is not a valid int to convert")
    else:
        if isinstance(ts_string, str) and ("," in ts_string):
            ts_string = ts_string.split(",")[0]
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts_string, fmt)
            except ValueError:
                continue
    raise ValueError(f"Time data '{ts_string}' does not match known formats")


def parse_qualifier(flag, data):
    """Helper to parse qualifiers across multiple types (regex, set)"""
    if isinstance(flag, str):
        return re.search(flag, data, flags=re.IGNORECASE)
    if isinstance(flag, set):
        return int(data) in flag
    return False

def gap_filter_batched(samples,  gap, timedeltas):
    """Helper to filter samples whose gap period is incompatible with forecast labels"""
    return [
        not any(
            (samples["end_timestamp"][i] + gap) > (samples["end_timestamp"][i] + time)
            for time in timedeltas.values()
        )
        for i in range(len(samples['end_timestamp']))
    ]
def main(
    month_deltas: list[int],
    matching_method=None,
    data_output_dir=None,
    gap=0,
):
    """
    Main method to create dataset for training and testing
    month_delta labels are applied in ascending order
    regardless of input
    """
    if not data_output_dir:
        data_output_dir = f"{OUTPUT_DIR}/{DATASET_DIR}"  # pylint: disable=E0602

    ensure_dir(data_output_dir)
    print("Output directories created:")
    print(f"  - Dataset: {data_output_dir}")

    pipeline = DataPipeline()
    samples = None
    if matching_method == "PSM":
        samples = pipeline.create_psm_samples()
    if not matching_method:
        samples = pipeline.create_regular_samples()
    dataset = Dataset.from_list(samples)
    df = pipeline.all_data
    del samples, pipeline
    month_label_to_deltas = {}
    for delta in sorted(month_deltas):
        month_label = f"{delta}_months"
        month_label_to_deltas[month_label] = relativedelta(months=delta)
    month_gap = relativedelta(months=gap) if gap > 0 else 0
    if month_gap > 0:
        dataset = dataset.filter(
                lambda samples: gap_filter_batched(samples, month_gap, month_label_to_deltas),
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
    dataset = dataset.map(
        lambda batch: preprocess_fn_mci(
            df,
            "pat_owner_id",
            batch,
            month_label_to_deltas,
            feature_to_regex,
            gap=month_gap,
        ),
        batched=True,
    )
    dataset.save_to_disk(data_output_dir)
    print(f"Labeled dataset saved to {data_output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create MCI dataset")
    parser.add_argument(
        "--month_deltas",
        nargs="+",
        type=int,
        default=[12, 24, 36],
        help="Month deltas for prediction windows",
    )
    parser.add_argument(
        "--matching_method",
        type=str,
        default=None,
        choices=["PSM", None],
        help="Matching method for samples",
    )
    parser.add_argument(
        "--gap",
        type=str,
        default=0,
        help="History / Qualifier gap in months",
    )

    parser.add_argument("--data_output_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        matching_method=args.matching_method,
        month_deltas=args.month_deltas,
        data_output_dir=args.data_output_dir,
        gap=args.month_gap,
    )
