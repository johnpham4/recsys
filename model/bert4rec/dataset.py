import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Optional
import random

class Bert4RecDataset(Dataset):
    """
    Dataset for BERT4Rec with masked language model training.

    Implements BERT-style masking strategy for sequential recommendation:
    - Randomly mask items in the sequence
    - Predict masked items based on context
    """

    def __init__(
        self,
        df,
        max_seq_len: int = 50,
        mask_prob: float = 0.15,
        padding_idx: int = 0,
        num_items: Optional[int] = None,
        for_inference: bool = False,
    ):
        """
        Args:
            df: Dataframe with 'item_sequence' column (already padded with -1)
            max_seq_len: Maximum sequence length
            mask_prob: Masking probability (only for training)
            padding_idx: Padding token index (use -1 for inference compatibility)
            num_items: Total number of items
            for_inference: If True, skip masking (for recommendation generation)
        """
        self.df = df
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.padding_idx = padding_idx
        self.num_items = num_items
        self.for_inference = for_inference

        # Extract sequences from df (already padded with -1)
        self.sequences = df['item_sequence'].values
        user_col = "user_indice" if "user_indice" in df.columns else "user_id"
        self.user_ids = df[user_col].values

    def _mask_sequence(self, sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply BERT-style masking to a sequence.

        Masking strategy (80-10-10 rule):
        - 80%: Replace with [MASK] token (we use padding_idx as mask)
        - 10%: Replace with random item
        - 10%: Keep original

        Args:
            sequence: [seq_len] - Input sequence

        Returns:
            Dictionary with:
                - input_ids: Masked sequence
                - labels: Original sequence (for loss calculation)
                - mask: Binary mask indicating which positions were masked
        """
        masked_sequence = sequence.copy()
        labels = sequence.copy()
        mask = np.zeros(len(sequence), dtype=np.int32)

        # Get valid (non-padding) positions
        # Check for -1 padding (from df), not padding_idx
        valid_positions = np.where(sequence != -1)[0]

        if len(valid_positions) == 0:
            # All padding, return as is
            return {
                'input_ids': masked_sequence,
                'labels': labels,
                'mask': mask
            }

        # Randomly select positions to mask
        num_to_mask = max(1, int(len(valid_positions) * self.mask_prob))
        mask_positions = np.random.choice(valid_positions, num_to_mask, replace=False)

        for pos in mask_positions:
            mask[pos] = 1
            rand = random.random()

            if rand < 0.8:
                # 80%: Replace with mask token (padding_idx)
                masked_sequence[pos] = self.padding_idx
            elif rand < 0.9:
                # 10%: Replace with random item
                if self.num_items is not None:
                    random_item = random.randint(0, self.num_items - 1)
                    masked_sequence[pos] = random_item
                else:
                    # If num_items not provided, use mask
                    masked_sequence[pos] = self.padding_idx
            else:
                # 10%: Keep original
                pass

        return {
            'input_ids': masked_sequence,
            'labels': labels,
            'mask': mask
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example with masking.

        Returns:
            Dictionary with:
                - input_ids: [max_seq_len] - Masked sequence (or original for inference)
                - labels: [max_seq_len] - Original sequence (target)
                - attention_mask: [max_seq_len] - Mask for padding (1 for real tokens)
        """
        sequence = self.sequences[idx]

        # For inference, skip masking
        if self.for_inference:
            input_ids = sequence
            labels = sequence
        else:
            # Apply masking for training
            masked_data = self._mask_sequence(sequence)
            input_ids = masked_data['input_ids']
            labels = masked_data['labels']

        # Keep -1 padding (will be mapped in model forward, like sequence model)
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (np.array(sequence) != -1).astype(np.int32)

        # Convert to tensors
        return {
            'user_ids': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


# Backward compatibility alias
BERT4RecDataset = Bert4RecDataset



def create_bert4rec_datasets(
    train_df,
    val_df,
    max_seq_len: int = 50,
    mask_prob: float = 0.15,
    padding_idx: int = 0,
    num_items: Optional[int] = None
) -> tuple:
    """
    Create training and validation datasets for BERT4Rec.

    Args:
        train_df: Training dataframe with 'item_sequence' column
        val_df: Validation dataframe with 'item_sequence' column
        max_seq_len: Maximum sequence length
        mask_prob: Masking probability
        padding_idx: Padding token index
        num_items: Total number of items

    Returns:
        (train_dataset, val_dataset)
    """
    train_dataset = BERT4RecDataset(
        df=train_df,
        max_seq_len=max_seq_len,
        mask_prob=mask_prob,
        padding_idx=padding_idx,
        num_items=num_items,
        for_inference=False
    )

    val_dataset = BERT4RecDataset(
        df=val_df,
        max_seq_len=max_seq_len,
        mask_prob=mask_prob,
        padding_idx=padding_idx,
        num_items=num_items,
        for_inference=False
    )

    return train_dataset, val_dataset
