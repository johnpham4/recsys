import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Positional embeddings for sequence order information."""

    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len]
        Returns:
            position_embeddings: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = x.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return self.position_embeddings(position_ids)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim, num_heads, dropout_prob=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, num_heads, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, seq_len, head_dim]

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.embed_dim)
        # [batch_size, seq_len, embed_dim]

        # Final linear projection
        output = self.out_linear(context)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_dim, ff_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout_prob)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout_prob)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x


class BERT4Rec(nn.Module):
    """
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers.

    Simplified PyTorch implementation adapted for recommendation systems.
    """

    def __init__(
        self,
        num_items,
        max_seq_len=50,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout_prob=0.1,
        padding_idx=0,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # Map -1 padding to padding_idx for embedding lookup
        # Embedding layer uses positive indices only
        self.item_embeddings = nn.Embedding(
            num_items + 1, embed_dim, padding_idx=padding_idx
        )
        self.positional_embeddings = PositionalEmbedding(max_seq_len, embed_dim)

        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_prob)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.masked_output = nn.Linear(embed_dim, num_items + 1, bias=False)
        self.masked_output.weight = self.item_embeddings.weight

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):

        padding_idx_tensor = torch.tensor(self.padding_idx, device=input_ids.device)
        input_ids = torch.where(input_ids == -1, padding_idx_tensor, input_ids)

        if attention_mask is None:
            attention_mask = (input_ids != self.padding_idx).long()

        item_embeds = self.item_embeddings(input_ids)  # [batch_size, seq_len, embed_dim]
        position_embeds = self.positional_embeddings(input_ids)  # [batch_size, seq_len, embed_dim]

        x = item_embeds + position_embeds
        x = self.dropout(x)

        # Create causal mask for bidirectional attention (allow full attention)
        # For BERT4Rec, we use bidirectional attention (not causal like GPT)
        # So we only need to mask padding tokens
        key_padding_mask = (attention_mask == 0)  # [batch_size, seq_len]

        # Pass through transformer layers
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=key_padding_mask)

        x = self.layer_norm(x)
        logits = self.masked_output(x)  # [batch_size, seq_len, num_items]

        return logits

    def predict_next_item(self, input_sequence, attention_mask=None):
        """
        Predict the next item given a sequence.

        Args:
            input_sequence: [batch_size, seq_len] - Item IDs in sequence
            attention_mask: [batch_size, seq_len] - Mask for padding

        Returns:
            next_item_logits: [batch_size, num_items] - Logits for next item prediction
        """
        # Get full sequence output
        logits = self.forward(input_sequence, attention_mask)

        # Use the last non-padding position for next item prediction
        if attention_mask is not None:
            # Find last valid position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            batch_indices = torch.arange(input_sequence.size(0), device=input_sequence.device)
            next_item_logits = logits[batch_indices, seq_lengths]  # [batch_size, num_items]
        else:
            # Use last position
            next_item_logits = logits[:, -1, :]  # [batch_size, num_items]

        return next_item_logits

    def recommend(self, input_sequence, k=10, attention_mask=None):
        self.eval()
        with torch.no_grad():
            next_item_logits = self.predict_next_item(input_sequence, attention_mask)
            next_item_scores = torch.softmax(next_item_logits, dim=-1)

            top_k_scores, top_k_items = torch.topk(next_item_scores, k, dim=-1)

        return top_k_items, top_k_scores
