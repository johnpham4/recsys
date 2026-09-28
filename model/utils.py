from tqdm.auto import tqdm
import pandas as pd
from typing import List
import numpy as np

def parse_dt(df, cols=["timestamp"]):
    return df.assign(
        **{
            col: lambda df: pd.to_datetime(df[col].astype(int), unit="ms")
            for col in cols
        }
    )

def handle_dtypes(df):
    return df.assign(rating=lambda df: df["rating"].astype(float))

def generate_negative_samples(
    df,
    user_col="user_indice",
    item_col="item_indice",
    label_col="rating",
    timestamp_col="timestamp",
    neg_label=0,
    neg_to_pos_ratio=1,
    seed=None,
    features: List[str] = [],
):
    # Set the random seed if provided for reproducibility.
    if seed is not None:
        np.random.seed(seed)

    # Calculate item popularity based on how frequently they appear in the DataFrame.
    item_popularity = df[item_col].value_counts()

    # Define all unique items from the DataFrame.
    items = item_popularity.index.values
    all_items_set = set(items)

    # Create a dictionary mapping each user to the set of items they interacted with.
    user_item_dict = df.groupby(user_col)[item_col].apply(set).to_dict()

    # Prepare popularity values for sampling probabilities.
    popularity = item_popularity.values.astype(np.float64)

    # Calculate item sampling probabilities proportional to their popularity.
    total_popularity = popularity.sum()
    if total_popularity == 0:
        # Handle edge case where no items have popularity by using uniform distribution.
        sampling_probs = np.ones(len(items)) / len(items)
    else:
        sampling_probs = popularity / total_popularity

    # Create a mapping from item to index to quickly access item-related data.
    item_to_index = {item: idx for idx, item in enumerate(items)}

    def generate_negative_samples_for_user(row):
        user = row[user_col]
        pos_items = user_item_dict[user]

        # Identify items not interacted with by the user (negative candidates).
        negative_candidates = all_items_set - pos_items
        num_neg_candidates = len(negative_candidates)

        if num_neg_candidates == 0:
            # If the user interacted with all items, skip this user.
            return []

        # The number of negative samples to generate equals the number of positive interactions, or fewer if there aren't enough candidates.
        num_neg = min(neg_to_pos_ratio, num_neg_candidates)

        # Convert the set of negative candidates to a list for indexing.
        negative_candidates_list = list(negative_candidates)

        # Obtain the indices and probabilities for the negative candidates.
        candidate_indices = [item_to_index[item] for item in negative_candidates_list]
        candidate_probs = sampling_probs[candidate_indices]
        candidate_probs /= candidate_probs.sum()  # Normalize probabilities.

        # Sample negative items for the user based on their probabilities.
        sampled_items = np.random.choice(
            negative_candidates_list, size=num_neg, replace=False, p=candidate_probs
        )

        return sampled_items

    tqdm.pandas()
    df_negative = (
        df.copy()
        .assign(
            negative_samples=lambda df: df.progress_apply(
                generate_negative_samples_for_user, axis=1
            ),
            **{label_col: neg_label}
        )
        .explode("negative_samples")
        .drop(columns=[item_col])
        .rename(columns={"negative_samples": item_col})[
            [user_col, item_col, label_col, timestamp_col, *features]
        ]
    )

    return df_negative


