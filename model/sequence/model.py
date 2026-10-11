import torch
import torch.nn as nn
from tqdm.auto import tqdm


class GRUModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, item_embedding=None, dropout=0.2):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items

        self.item_embedding = item_embedding or nn.Embedding(
            n_items + 1, embedding_dim, padding_idx=n_items
        )

        self.user_embeddings = nn.Embedding(n_users, embedding_dim=embedding_dim)

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.relu,
            self.dropout,
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids, target_item, sequence):

        padding_idx_tensor = torch.tensor(self.item_embedding.padding_idx, device=sequence.device)
        input_seq = torch.where(sequence == -1, padding_idx_tensor, sequence)
        target_item = torch.where(target_item == -1, padding_idx_tensor, target_item)

        embed_seq = self.item_embedding(input_seq)                # [B, seq_len, D]
        _, hs = self.gru(embed_seq)                               # [1, B, D]
        gru_output = hs.squeeze(0)                                # [B, D]
        embed_target = self.item_embedding(target_item)           # [B, D]
        embed_user = self.user_embeddings(user_ids)               # [B, D]
        embed_combined = torch.cat((gru_output, embed_target, embed_user), dim=1)
        outputs = self.fc(embed_combined)                          # [B, 1]
        return outputs

    def predict(self, user, item_sequence, target_item):
        return self.forward(user, target_item, item_sequence)

    def recommend(self, users, item_sequences, k, batch_size=128):
        self.eval()
        all_items = torch.arange(self.n_items, device=users.device)
        recs, user_indices, scores = [], [], []

        with torch.no_grad():
            total_users = users.size(0)
            for i in tqdm(range(0, total_users, batch_size), desc="Generating recommendations"):
                user_batch = users[i : i + batch_size]
                item_sequence_batch = item_sequences[i : i + batch_size]

                user_batch_expand = user_batch.unsqueeze(1).expand(-1, len(all_items)).reshape(-1)
                items_batch = all_items.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)
                item_sequence_batch_expand = item_sequence_batch.unsqueeze(1).repeat(1, len(all_items), 1)
                item_sequence_batch_expand = item_sequence_batch_expand.view(-1, item_sequence_batch.size(-1))

                batch_scores = self.predict(user_batch_expand, item_sequence_batch_expand, items_batch)
                batch_scores = batch_scores.view(len(user_batch), -1)

                topk_scores, topk_indices = torch.topk(batch_scores, k, dim=1)
                topk_items = all_items[topk_indices]

                user_indices.extend(user_batch.repeat_interleave(k).cpu().tolist())
                recs.extend(topk_items.cpu().flatten().tolist())
                scores.extend(topk_scores.cpu().flatten().tolist())

        return {
            "user_indice": user_indices,
            "recommendation": recs,
            "score": scores,
        }


SequenceModel = GRUModel

