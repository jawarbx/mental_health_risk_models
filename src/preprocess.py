"""Dataset creation script"""

import argparse
import json
import math
import os
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from transformers import AutoTokenizer

from data_pipeline import DataPipeline
from tqdm import tqdm
tqdm.pandas()

SCRIPT_DIR = Path(__file__).parent.resolve()
dotenv_path = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DATASET_DIR = os.getenv("DATASET_DIR")
MCI_QA_MEDKEY_PATH = os.getenv("MCI_QA_MEDKEY_PATH")
MCI_ICD_REGEX = os.getenv("MCI_ICD_REGEX")
MCI_MED_REGEX = os.getenv("MCI_MED_REGEX")
#Sanity check using manual date extraction
MCI_DATES_PATCH = os.getenv("MCI_DATES_PATCH")
MCI_DATES_DF = pd.read_csv(MCI_DATES_PATCH)

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


def label_fn_mci_forecast(
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
                    and bounds["end_time"] >= d[timestamp] >= bounds["start_time"]
                ]
                label_hits = [
                    d for d in filtered_history if parse_qualifier(regex, d[data])
                ]
            if delta_name not in labels:
                labels[delta_name] = []
            labels[delta_name].append(any(label_hits))
    final_labels = {delta_name: any(checks) for delta_name, checks in labels.items()}
    label_vector = [int(final_labels[name]) for name in sorted(final_labels.keys())]
    return label_vector

# Full message history method:
# Given some gap M,
# For each message history:
#   1. find the first instance of the qualifier, mark the end_date = (date of the qualifier - M)
#   2. If end_date <= start date date of the history, label it 1 and use the entire message history
#      ONLY in the training set
#   3. Else, use the message history up to the end_date defined in 1
#   4. If the message history never has a qualifier, mark it 0 and use the entire message history

#Lookback method implemented with static date list for sanity check
def create_and_label_mci_lookback_samples_patch(
        df: pd.DataFrame,
        gap: int,
        earliest_sign_df: pd.DataFrame
):
    month_gap = relativedelta(months=gap) if gap > 0 else 0
    df = df.copy()
    df["sorted_message_histories"] = df["sorted_message_histories"].progress_apply(
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
    earliest_sign_df['earliest_sign'] = earliest_sign_df['earliest_sign'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df = df.merge(earliest_sign_df[['pat_owner_id','earliest_sign']],on='pat_owner_id', how='left')
    df = df.rename(columns={"earliest_sign":"_first_qualifying_date"})
    df = df.progress_apply(lambda row: apply_lookback(row, month_gap), axis=1)
    df = df[
        df["sorted_message_histories"].progress_apply(lambda h: isinstance(h, list) and len(h) > 0)
    ].reset_index(drop=True)

    return df

#TODO: Fix parsing here
def create_and_label_mci_lookback_samples(
        df: pd.DataFrame,
        gap: int,
        feature_to_regex:dict,
):
    """
    Method to return MCI labeled dataframe based on lookback method.
    """
    month_gap = relativedelta(months=gap) if gap > 0 else 0
    df = df.copy()

    for feature, (qualifier, ts_key, _) in feature_to_regex.items():
        df[feature] = df[feature].progress_apply(
            lambda h: (
                [
                    {**d, ts_key: parse_timestamp(d[ts_key])}
                    for d in h
                    if d is not None
                ]
                if isinstance(h, list)
                else h
            )
        )

    df["sorted_message_histories"] = df["sorted_message_histories"].progress_apply(
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

    df["_first_qualifying_date"] = df.apply(lambda row: first_qualifying_date(row, feature_to_regex), axis=1)
    df = df.progress_apply(lambda row: apply_lookback(row, month_gap), axis=1)

    df = df[
        df["sorted_message_histories"].progress_apply(lambda h: isinstance(h, list) and len(h) > 0)
    ].reset_index(drop=True)

    return df

def preprocess_fn_mci_lookback(
    df,
    id_feature,
    gap,
    feature_to_regex
):
    df = create_and_label_mci_lookback_samples_patch(df, gap, MCI_DATES_DF)
    df["_text"] = df["sorted_message_histories"].apply(
        lambda h: "\n".join(msg["content"] for msg in h if msg is not None)
    )
    tokenized = tokenizer(
        df["_text"].tolist(),
        truncation=False,
        padding=False,
    )

    df["input_ids"]      = tokenized["input_ids"]
    df["attention_mask"] = tokenized["attention_mask"]

    if "token_type_ids" in tokenized:
        df["token_type_ids"] = tokenized["token_type_ids"]

    # ------------------------------------------------------------------
    # 4. Build and return the Dataset from the enriched DataFrame
    # ------------------------------------------------------------------
    output_cols = [
        id_feature,
        "label",
        "_train_only",
        "input_ids",
        "attention_mask",
    ]

    return Dataset.from_dict(df[output_cols].to_dict(orient="list"))


def preprocess_fn_mci_forecast(
    df,
    id_feature,
    samples: dict,
    timedeltas: dict[str, str],
    feature_to_regex: dict[str, str],
    gap=0,
):
    """Batched method for labeling dataset and encoding"""
    label_vectors = [
        label_fn_mci_forecast(
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
    tokenized = tokenizer(
        samples["content"],
        truncation=False,
        padding=False,
    )

    tokenized["labels"] = label_vectors

    return tokenized


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


def gap_filter_batched(samples, gap, timedeltas):
    """Helper to filter samples whose gap period is incompatible with forecast labels"""
    return [
        not any(
            (parse_timestamp(samples["end_timestamp"][i]) + gap)
            > (parse_timestamp(samples["end_timestamp"][i]) + time)
            for time in timedeltas.values()
        )
        for i in range(len(samples["end_timestamp"]))
    ]

def first_qualifying_date(row,feature_to_regex):
    """Helper to find the first qualifying date given a timestamped feature list"""
    candidates = []
    for feature, (qualifier, ts_key, val_key) in feature_to_regex.items():
        history = row[feature]
        if not isinstance(history, list):
            continue
        for entry in history:
            if entry is None:
                continue
            # qualifier is either a regex string or a set of ids
            if parse_qualifier(qualifier, entry.get(val_key, "")):
                ts = entry.get(ts_key)
                if ts is not None:
                    candidates.append(ts)
    return min(candidates) if candidates else None

def apply_lookback(row, gap_delta):
    """Helper to apply lookback method to patient row"""
    first_date = row["_first_qualifying_date"]
    history    = row["sorted_message_histories"]
    history_start = (
        history[0]["timestamp"] if history else None
    )

    # Rule 4: never qualifies → label=0, full history
    if (first_date is None) or (first_date is pd.NaT) or (not isinstance(first_date, datetime) and pd.isna(first_date)):
        row["label"]       = 0
        row["_train_only"] = False
        return row

    try:
        end_date = first_date - gap_delta
    except Exception as e:
        print(first_date)
        print(gap_delta)
        raise e

    # Rule 5: end_date at or before history start → label=1, full history, train only
    if history_start is None or end_date <= history_start:
        row["label"]       = 1
        row["_train_only"] = True
        return row

    # Rule 6: truncate all histories to end_date → label=1
    row["sorted_message_histories"] = [
        m for m in history
        if m["timestamp"] < end_date
    ]
    row["label"]       = 1
    row["_train_only"] = False
    return row

def create_lookback_samples(
    pipeline=None,
    lookback_gap=None,
    data_output_dir=None,
    feature_to_regex=None
):
    print("Using lookback method!")
    dataset = preprocess_fn_mci_lookback(pipeline.all_data, 'pat_owner_id', gap=lookback_gap, feature_to_regex=feature_to_regex)
    dataset.save_to_disk(data_output_dir)
    print(f"Labeled lookback dataset saved to {data_output_dir}")

def main(
    month_deltas: list[int],
    matching_method=None,
    sample_creation_method=None,
    lookback_gap=None,
    forecast_gap=None,
    data_output_dir=None,
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

    if sample_creation_method == "lookback":
        create_lookback_samples(pipeline=pipeline, lookback_gap=lookback_gap, data_output_dir=data_output_dir, feature_to_regex=feature_to_regex)
    if sample_creation_method == "forecast":
        create_forecast_samples(pipeline=pipeline, matching_method=matching_method, data_output_dir=data_output_dir, forecast_gap=forecast_gap, month_deltas=month_deltas, feature_to_regex=feature_to_regex)
    else:
        print("Please choose a method to create samples!")
        exit(1)

def create_forecast_samples(
    pipeline=None,
    month_deltas=None,
    matching_method=None,
    data_output_dir=None,
    forecast_gap=None,
    feature_to_regex=None,
):
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
    month_gap = relativedelta(months=forecast_gap) if forecast_gap > 0 else 0
    if month_gap > 0:
        dataset = dataset.filter(
            lambda samples: gap_filter_batched(
                samples, month_gap, month_label_to_deltas
            ),
            batched=True,
            batch_size=10000,
        )

    for feature in feature_to_regex.keys():
        df[feature] = df[feature].apply(
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
        lambda batch: preprocess_fn_mci_forecast(
            df_grouped,
            "pat_owner_id",
            batch,
            month_label_to_deltas,
            feature_to_regex,
            gap=month_gap,
        ),
        batched=True,
    )
    dataset.save_to_disk(data_output_dir)
    print(f"Labeled forecast dataset saved to {data_output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create MCI dataset")
    parser.add_argument(
        "--month_deltas",
        nargs="+",
        type=int,
        default=[18, 24, 36],
        help="month deltas for prediction windows using forecast method",
    )
    parser.add_argument(
        "--matching_method",
        type=str,
        default=None,
        choices=["PSM", None],
        help="Matching method for samples",
    )

    parser.add_argument(
        "--sample_creation_method",
        type=str,
        default=None,
        choices=["forecast", 'lookback'],
        help="Creation method for samples",
    )
    parser.add_argument(
        "--forecast_gap",
        type=int,
        default=12,
        help="History / Qualifier gap in months for forecast method",
    )

    parser.add_argument(
        "--lookback_gap",
        type=int,
        default=36,
        help="Lookback gap in months for lookback method"
    )

    parser.add_argument("--data_output_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        matching_method=args.matching_method,
        month_deltas=args.month_deltas,
        data_output_dir=args.data_output_dir,
        sample_creation_method=args.sample_creation_method,
        forecast_gap=args.forecast_gap,
        lookback_gap=args.lookback_gap,
    )
