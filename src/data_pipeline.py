"""
General data pipeline for MCI and Depression Model Training
Implements sliding window based sample generation
"""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class DataPipeline:
    """Data pipeline class"""

    def __init__(
        self,
        model_name=None,
        min_tokens=128,
        message_data_path=None,
        icd_data_path=None,
        demo_data_path=None,
        med_data_path=None,
    ):
        """
        Data pipeline initialization
        """

        self.model_name = model_name
        self.min_tokens = min_tokens
        # Read in raw dataframes
        all_data = self.__safe_read(message_data_path)
        if not all_data:
            print("Message data could not load, unable to initialize data pipeline")
            return None
        icd_df = self.__safe_read(icd_data_path)
        demo_df = self.__safe_read(demo_data_path)
        med_df = self.__safe_read(med_data_path)

        if not icd_df or not demo_df or not med_df:
            print(
                "Need at least one extra dataframe to phenotype, unable to initlize data pipeline"
            )
            return None

        # Merge by pt_id
        all_data = (
            pd.merge(all_data, icd_df, on="pat_owner_id", how="outer")
            if icd_df
            else all_data
        )
        all_data = (
            pd.merge(all_data, demo_df, on="pat_owner_id", how="outer")
            if demo_df
            else all_data
        )
        all_data = (
            pd.merge(all_data, med_df.transpose(), on="pat_owner_id", how="outer")
            if med_df
            else all_data
        )

        # Remove pts without messages and clean copies
        self.all_data = all_data[
            (all_data.pat_to_doctor_msgs > 0) & (all_data.pat_to_doctor_total_words > 0)
        ]
        del icd_df, demo_df, med_df, all_data
        # Done with init!

    def psm_split(self, train_split=0.8, test_split=0.15, val_split=0.05):
        """
        Method to generate train, test, validation set using propensity score matched
        patient histories.
        1. Run PSM method on entire population
        2. Run sample generation on PSM'd population
        3. Create training, validation, and testing splits
        """
        new_population = self.psm(self.all_data)
        samples = self.sample_generation(
            new_population, "pat_owner_id", "refined_pat_encounters"
        )
        training, testing = train_test_split(samples, test_size=1 - train_split)
        testing, validation = train_test_split(
            testing, test_size=(test_split / (test_split + val_split))
        )
        return training, testing, validation

    def psm(self, df):
        """Method to calculate propensity scores row wise and match rows with them"""
        return df

    def sample_generation(self, df, id_attribute, sequence_attribute, num_workers=4):
        """Method to generate samples from dataframe df from each row in parallel"""
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    DataPipeline.generate_windows,
                    row[id_attribute],
                    row[sequence_attribute],
                    self.min_tokens,
                    self.model_name,
                )
                for _, row in df.iterrows()
            ]
            results = [f.result() for f in futures]
            return results

    @staticmethod
    def init_worker(model_name):
        """Initializer for window generation workers"""
        global tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def get_item_token_counts(seq):
        """
        Helper to get token counts of each item in a sequence
        Assumes the items have the key "content"
        """
        global tokenizer
        contents = [item["content"] for item in seq]
        enc = tokenizer(
            contents, add_special_tokens=False, padding=False, truncation=False
        )
        return [len(ids) for ids in enc["input_ids"]]

    @staticmethod
    def generate_windows(sample_id, message_list, min_tokens, model_name):
        """
        Method to generate window sample from an sequence message_list, 
        ordered from most to least recent
        1. The entire sequence is tokenized per message and total tokens per message are calculated
        2. From the entire sequence, a decreasing window is used to generate samples the last
        sample < min_tokens
        3. The samples are returned for later processing
        """
        DataPipeline.init_worker(model_name)
        token_counts = DataPipeline.get_item_token_counts(message_list)
        samples = list(
            DataPipeline.decreasing_token_limited_windows(
                sample_id, message_list, token_counts, min_tokens
            )
        )

        return samples

    @staticmethod
    def decreasing_token_limited_windows(seq_id, seq, token_counts, min_tokens):
        """
        Method to generate samples from a sequence while the min_tokens condition is satisfied
        """
        windows = []
        running = sum(token_counts)
        for i in range(len(seq), 0, -1):
            if running >= min_tokens:
                window = seq[:i]
                windows.append(
                    {
                        "seq_id": seq_id,
                        "content": "\n".join([item["content"] for item in window]),
                        "start_timestamp": window[0]["timestamp"],
                        "end_timestamp": window[-1]["timestamp"],
                        "num_tokens": running,
                    }
                )
            running -= token_counts[i - 1]
        return windows

    def __safe_read(self, path):
        """
        Helper to safely read from paths (json, csv supported only)
        """
        if ".json" in path:
            try:
                return pd.read_json(path)
            except FileNotFoundError:
                print(f"Could not find: {path}")
            except Exception as e:
                print(f"Unexpected exception occurred: {e}")
            return None
        if ".csv" in path:
            try:
                return pd.read_csv(path)
            except FileNotFoundError:
                print(f"Could not find: {path}")
            except Exception as e:
                print(f"Unexpected exception occurred: {e}")
            return None
