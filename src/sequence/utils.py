import numpy as np

def generate_item_sequences(
    df,
    user_col,
    item_col,
    timestamp_col,
    sequence_length,
    padding=True,
    padding_value=-1,
):
    def get_item_sequence(sub_df):
        sequences = []
        for i in range(len(sub_df)):
            # Get item indices up to the current row (excluding the current row)
            sequence = sub_df.iloc[:i].tolist()[-sequence_length:]
            if padding:
                padding_needed = sequence_length - len(sequence)
                sequence = np.pad(
                    sequence,
                    (padding_needed, 0),  # Add padding at the beginning
                    "constant",
                    constant_values=padding_value,
                )
            sequences.append(sequence)
        return sequences

    df = df.sort_values(timestamp_col)
    df["item_sequence"] = df.groupby(user_col, group_keys=True)[item_col].transform(
        get_item_sequence
    )
    df["item_sequence"] = df["item_sequence"].fillna("").apply(list)

    return df