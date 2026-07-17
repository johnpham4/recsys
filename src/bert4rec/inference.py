import torch
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from .model import BERT4Rec


class BERT4RecInference:
    """
    Inference wrapper for BERT4Rec model.
    
    Handles loading the model and generating recommendations.
    """
    
    def __init__(
        self,
        model: BERT4Rec,
        device: str = "cpu",
        max_seq_len: int = 50,
        padding_idx: int = 0
    ):
        """
        Args:
            model: Trained BERT4Rec model
            device: Device to run inference on
            max_seq_len: Maximum sequence length
            padding_idx: Padding token index
        """
        self.model = model
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"BERT4Rec inference initialized on device: {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        max_seq_len: int = 50,
        padding_idx: int = 0
    ):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            max_seq_len: Maximum sequence length
            padding_idx: Padding token index
        
        Returns:
            BERT4RecInference instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model hyperparameters
        if 'hyperparameters' in checkpoint:
            hparams = checkpoint['hyperparameters']
            model = BERT4Rec(
                num_items=hparams['num_items'],
                max_seq_len=hparams['max_seq_len'],
                embed_dim=hparams['embed_dim'],
                num_layers=hparams['num_layers'],
                num_heads=hparams['num_heads'],
                ff_dim=hparams['ff_dim'],
                dropout_prob=hparams.get('dropout_prob', 0.1),
                padding_idx=padding_idx
            )
        else:
            # Fallback: try to load directly
            model = BERT4Rec(
                num_items=10000,  # Default, should be overridden
                max_seq_len=max_seq_len,
                padding_idx=padding_idx
            )
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model, device, max_seq_len, padding_idx)
    
    def _prepare_sequence(
        self, 
        sequence: List[int],
        truncate: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare sequence for inference.
        
        Args:
            sequence: List of item IDs
            truncate: Whether to truncate if sequence is too long
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Convert to numpy array
        sequence_array = np.array(sequence, dtype=np.int64)
        
        # Truncate or pad
        if len(sequence_array) > self.max_seq_len:
            if truncate:
                sequence_array = sequence_array[-self.max_seq_len:]
            else:
                raise ValueError(f"Sequence length {len(sequence_array)} exceeds max_seq_len {self.max_seq_len}")
        elif len(sequence_array) < self.max_seq_len:
            # Pad with padding_idx
            sequence_array = np.pad(
                sequence_array, 
                (self.max_seq_len - len(sequence_array), 0), 
                constant_values=self.padding_idx
            )
        
        # Create attention mask
        attention_mask = (sequence_array != self.padding_idx).astype(np.int32)
        
        return {
            'input_ids': torch.tensor(sequence_array, dtype=torch.long).unsqueeze(0).to(self.device),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(self.device)
        }
    
    def predict_next_item(
        self,
        sequence: List[int],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Predict the next item for a given sequence.
        
        Args:
            sequence: List of item IDs representing user's interaction history
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with:
                - predicted_items: List of top-k item IDs
                - predicted_scores: List of top-k scores
                - all_scores: Full scores for all items (optional)
        """
        with torch.no_grad():
            # Prepare input
            inputs = self._prepare_sequence(sequence)
            
            # Get predictions
            next_item_logits = self.model.predict_next_item(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            # Convert to probabilities
            next_item_probs = torch.softmax(next_item_logits, dim=-1)
            
            # Get top-k
            top_k_scores, top_k_items = torch.topk(next_item_probs, top_k, dim=-1)
            
            return {
                'predicted_items': top_k_items.squeeze(0).cpu().tolist(),
                'predicted_scores': top_k_scores.squeeze(0).cpu().tolist(),
                'all_scores': next_item_probs.squeeze(0).cpu().tolist()
            }
    
    def recommend_batch(
        self,
        sequences: List[List[int]],
        k: int = 10,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for multiple sequences in batch.
        
        Args:
            sequences: List of sequences (each is a list of item IDs)
            k: Number of recommendations per sequence
            batch_size: Batch size for inference
        
        Returns:
            List of recommendation dictionaries
        """
        all_recommendations = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Prepare batch inputs
            batch_input_ids = []
            batch_attention_masks = []
            
            for seq in batch_sequences:
                prepared = self._prepare_sequence(seq)
                batch_input_ids.append(prepared['input_ids'].squeeze(0))
                batch_attention_masks.append(prepared['attention_mask'].squeeze(0))
            
            batch_input_ids = torch.stack(batch_input_ids).to(self.device)
            batch_attention_masks = torch.stack(batch_attention_masks).to(self.device)
            
            with torch.no_grad():
                # Get recommendations
                top_k_items, top_k_scores = self.model.recommend(
                    batch_input_ids,
                    k=k,
                    attention_mask=batch_attention_masks
                )
                
                # Convert to lists
                for j in range(len(batch_sequences)):
                    all_recommendations.append({
                        'predicted_items': top_k_items[j].cpu().tolist(),
                        'predicted_scores': top_k_scores[j].cpu().tolist()
                    })
        
        return all_recommendations
    
    def get_sequence_embedding(
        self,
        sequence: List[int]
    ) -> np.ndarray:
        """
        Get the embedding representation of a sequence.
        
        Useful for:
        - Similarity search between sequences
        - Clustering user behaviors
        - Feature engineering for downstream tasks
        
        Args:
            sequence: List of item IDs
        
        Returns:
            Sequence embedding as numpy array
        """
        with torch.no_grad():
            inputs = self._prepare_sequence(sequence)
            
            # Get model output (before final projection)
            # We need to modify this in the model to expose the sequence output
            # For now, we'll use the item embeddings averaged
            
            item_embeds = self.model.item_embeddings(inputs['input_ids'])
            position_embeds = self.model.positional_embeddings(inputs['input_ids'])
            
            # Combine embeddings
            x = item_embeds + position_embeds
            
            # Pass through transformer layers
            attention_mask = (inputs['attention_mask'] == 0)
            for transformer_block in self.model.transformer_blocks:
                x = transformer_block(x, mask=attention_mask)
            
            # Apply layer norm
            x = self.model.layer_norm(x)
            
            # Average pooling over sequence (ignoring padding)
            mask_expanded = inputs['attention_mask'].unsqueeze(-1).float()
            summed = (x * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            sequence_embedding = (summed / counts).squeeze(0)
            
            return sequence_embedding.cpu().numpy()
    
    def compute_sequence_similarity(
        self,
        sequence1: List[int],
        sequence2: List[int]
    ) -> float:
        """
        Compute cosine similarity between two sequences.
        
        Args:
            sequence1: First sequence
            sequence2: Second sequence
        
        Returns:
            Cosine similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        embedding1 = self.get_sequence_embedding(sequence1).reshape(1, -1)
        embedding2 = self.get_sequence_embedding(sequence2).reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0, 0]
        return float(similarity)
