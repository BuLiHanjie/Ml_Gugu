import numpy as np
import torch
import torch.nn as nn

# from torch_model.torch_args.torch_param import EmbeddingParam


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = _weight
        self.emb = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            _weight=self._weight
        )

    def forward(self, x):
        return self.emb(x)


class ContinuousEmbedding(nn.Module):

    def __init__(self, embedding_dim, bins, mode='linear', padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None):
        super().__init__()
        self.low = nn.Parameter(torch.tensor([-np.inf] + bins))
        self.high = nn.Parameter(torch.tensor(bins + [np.inf]))
        self.num_embeddings = len(bins) + 1
        self.mode = mode
        self.emb = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight
        )

    def forward(self, x: torch.Tensor):
        arange = torch.arange(0, self.num_embeddings, device=x.device)
        x = x.unsqueeze(-1).repeat((1,) * len(x.shape) + (self.low.shape[-1],))
        s = (x > self.low) * (x <= self.high)
        index = s.argmax(-1)
        diff = torch.abs(index.unsqueeze(-1) - arange)
        if self.mode == 'linear':
            scale = 1 / (diff + 1).float()
        else:
            scale = None
        embedding = self.emb(arange)
        e = (scale.unsqueeze(-1) * embedding).sum(-2)
        return e
