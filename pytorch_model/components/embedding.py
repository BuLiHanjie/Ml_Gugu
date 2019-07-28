import torch
import torch.nn as nn

from torch_model.torch_args.torch_param import EmbeddingParam


class Embedding(nn.Module):

    def __init__(self, param: EmbeddingParam):
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=param.num_embeddings,
            embedding_dim=param.embedding_dim,
            padding_idx=param.padding_idx,
            max_norm=param.max_norm,
            norm_type=param.norm_type,
            scale_grad_by_freq=param.scale_grad_by_freq,
            sparse=param.sparse,
            _weight=param._weight
        )

    def forward(self, x):
        return self.emb(x)