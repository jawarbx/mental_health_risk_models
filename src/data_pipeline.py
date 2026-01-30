"""
General data pipeline for MCI and Depression Model Training
Implements sliding window based sample generation
"""

from concurrent.futures import ProcessPoolExecutor

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer
from xgboost import DMatrix, XGBClassifier

load_dotenv()


class DataPipeline:
    """Data pipeline class"""

    def __init__(
        self,
        model_name=MODEL_NAME, # pylint: disable=E0602
        min_tokens=128,
    ):
        """
        Data pipeline initialization
        """

        self.model_name = model_name
        self.min_tokens = min_tokens
        # Read in raw dataframes
        all_data = self.__safe_read(PT_MESSAGES_PATH)  # pylint: disable=E0602
        if not all_data:
            print("Message data could not load, unable to initialize data pipeline")
        icd_df = self.__safe_read(PT_ICDS_PATH)  # pylint: disable=E0602
        demo_df = self.__safe_read(PT_DEMO_PATH)  # pylint: disable=E0602
        med_df = self.__safe_read(PT_MED_PATH)  # pylint: disable=E0602
        qa_df = self.safe_read(PT_QA_MEDS_DX_PBL_PATH) # pylint: disable=E0602
        if not icd_df or not demo_df or not med_df:
            print(
                "Need at least one extra dataframe to phenotype, unable to initlize data pipeline"
            )

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
            pd.merge(all_data, med_df, on="pat_owner_id", how="outer")
            if med_df
            else all_data
        )
        
        all_data = (
            pd.merge(all_data, qa_df, on="pat_owner_id", how="outer")
            if qa_df 
            else all_data
        )

        # Remove pts without messages and clean copies
        self.all_data = all_data[
            (all_data.pat_to_doctor_msgs > 0) & (all_data.pat_to_doctor_total_words > 0)
        ]
        del icd_df, demo_df, med_df, all_data, qa_df
        # Done with init!

    def create_regular_samples(self):
        """Method to return general population samples"""
        samples = self.sample_generation(df, "pat_owner_id", "refined_pat_encounters")
        return samples

    def create_psm_samples(self, ratio=1):
        """
        Method to create propensity score matched patient population.
        1. Run PSM method on entire population
        2. Run sample generation on PSM'd population
        """
        new_population = self.psm(self.all_data,ratio)
        samples = self.sample_generation(
            new_population, "pat_owner_id", "refined_pat_encounters"
        )
        return samples

    def psm(self, df, label="label",ratio=1):
        """Method to calculate propensity scores row wise and match rows with them
        Assumes df is labeled with treated patients already with label"""
        df["icd_set"] = df["icd_maps"].apply(lambda x: set(map(lambda y: y[1], x)))
        df["med_set"] = df["encounters"].apply(
            lambda x: set(
                sum(
                    list(
                        map(
                            lambda y: list(
                                map(
                                    str.strip,
                                    str.strip(y["med_description"]).split(", "),
                                )
                            ),
                            x,
                        )
                    ),
                    [],
                )
            )
        )

        med_set_list = sorted(list(set.union(*list(df.med_set))))
        med_set_map = {med_id: index for index, med_id in enumerate(med_set_list)}

        icd_set_list = sorted(list(set.union(*list(df.dx_list))))
        icd_set_map = {dx_id: index for index, dx_id in enumerate(icd_set_list)}

        med_set_map_len = len(med_set_map.keys())
        icd_set_map_len = len(icd_set_map.keys())

        num_features_med = len(
            self.__map_ids_to_bytestring(df.med_set[1], med_set_map, med_set_map_len)
        )
        x_sparse_med = self.__stream_sets_to_sparse(
            df.med_set, num_features_med, med_set_map, med_set_map_len
        )

        num_features_icd = len(
            self.__map_ids_to_bytestring(df.icd_set[1], icd_set_map, icd_set_map_len)
        )
        x_sparse_icd = self.__stream_sets_to_sparse(
            df.icd_set, num_features_icd, icd_set_map, icd_set_map_len
        )

        x_race_eth_sex = (
            df[["race", "ethnicity", "sex"]]
            .astype("category")
            .apply(lambda x: x.cat.codes)
            .values
        )
        age_locations = df["age"]

        x_combined = sp.hstack(
            [
                age_locations.values.reshape(-1, 1),
                x_race_eth_sex,
                x_sparse_med,
                x_sparse_icd,
            ]
        )

        feature_types = (
            ["q"]
            + ["c"] * x_race_eth_sex.shape[1]
            + ["c"] * x_sparse_med.shape[1]
            + ["c"] * x_sparse_icd.shape[1]
        )

        dtrain = DMatrix(x_combined)
        dtrain.set_info(feature_types=feature_types)

        labels = df[label].values

        neg, pos = np.bincount(labels)
        scale_pos_weight = neg / pos
        model = XGBClassifier(
            objective="binary:logistic",
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            max_delta_step=1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            enable_categorical=True,
            device="cuda",
        )
        model.fit(x_combined, labels)

        propensity_scores = model.predict_proba(x_combined)[:, 1]
        df["propensity_scores"] = pd.DataFrame(propensity_scores)
        treated = df[df[label] == 1]
        control = df[df[label] == 0]

        treated_scores = treated[["propensity_scores"]].values.astype("float32")
        control_scores = control[["propensity_scores"]].values.astype("float32")

        index = faiss.IndexFlatL2(1)
        index.add(control_scores)

        batch_size = 5000
        all_distances = []
        all_indices = []

        for i in range(0, len(treated_scores), batch_size):
            distances, indices = index.search(treated_scores[i : i + batch_size], ratio)

            all_distances.append(distances)
            all_indices.append(indices)

            print(f"Processed {i + batch_size} treated samples")

        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)

        print("KNN Matching completed successfully!")

        matched_control = control.iloc[all_indices.flatten()]

        matched_data = pd.concat([treated, matched_control])

        return df[df["pat_owner_id"].isin(matched_data["pat_owner_id"])]

    def __map_ids_to_bytestring(self, id_set, id_map, id_map_len):
        bytearray_size = (id_map_len + 7) // 8  # Round up to the nearest byte
        bytearray_data = bytearray(bytearray_size)

        for e in id_set:
            # Find the mapped index of the ID and set that bit to 1
            index = id_map[e]
            byte_index = index // 8  # Which byte
            bit_index = index % 8  # Which bit within the byte
            bytearray_data[byte_index] |= 1 << bit_index  # Set the bit to 1

        return bytearray_data

    def __stream_sets_to_sparse(self, feature_col, num_features, id_map, id_map_len):
        """Convert bytearrays to a sparse CSR matrix sequentially."""
        duration = len(feature_col)
        row_indices = []
        col_indices = []
        data_values = []
        progress_bar = tqdm(
            enumerate(feature_col), total=duration, position=0, leave=True
        )
        for row_idx, id_set in progress_bar:
            byte_arr = self.__map_ids_to_bytestring(id_set, id_map, id_map_len)
            indices = np.nonzero(byte_arr)[0]  # Nonzero indices
            values = np.array(byte_arr)[indices]  # Nonzero values

            row_indices.extend([row_idx] * len(indices))
            col_indices.extend(indices)
            data_values.extend(values)

        # Create sparse matrix efficiently
        sparse_matrix = sp.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(len(feature_col), num_features),
        )

        return sparse_matrix

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
                        "pat_owner_id": seq_id,
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
        try:
            if ".json" in path:
                return pd.read_json(path)
            elif ".csv" in path:
                return pd.read_csv(path)
            else:
                return None
        except FileNotFoundError:
            print(f"Could not find: {path}")
            return None
        except Exception as e:
            print(f"Unexpected exception occurred: {e}")
            return None
