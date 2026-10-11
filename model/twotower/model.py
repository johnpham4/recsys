import torch
import torch.nn
from tqdm.auto import tqdm


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, max_len):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items

        