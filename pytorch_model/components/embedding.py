import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DynEmbedding(nn.Module):
    def __init__(self, embedding_dim, optimizer, emb_step=1000, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, device=None):
        super().__init__()
        self.optimizer = optimizer
        self.embedding_dim = embedding_dim
        self.emb_step = emb_step
        # self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.device = device
        self.params = list()
        self.index_map = dict()
        self.cnt = 0

        if padding_idx is not None:
            self.cnt = 1
            self.add_param()
            self.index_map[padding_idx] = 0

    def add_param(self):
        self.params.append(
            Embedding(
                self.emb_step,
                self.embedding_dim,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse
            ).to(self.device))
        self.add_module('params_{}'.format(len(self.params) - 1), self.params[-1])
        # self.register_parameter('params_{}'.format(len(self.params) - 1), self.params[-1])
        self.optimizer.add_param(self.params[-1].parameters())

    def forward(self, x):
        _shape = x.shape
        res = list()
        for _x in x.reshape(-1):
            # _x = int(_x)
            if _x not in self.index_map:
                if self.training:
                    self.index_map[_x] = self.cnt
                    self.cnt += 1
                    if self.cnt % self.emb_step == 1:
                        self.add_param()
                    index = self.index_map[_x]
                    res.append(self.params[index // self.emb_step](torch.tensor(index % self.emb_step, device=self.device)))
                else:
                    res.append(torch.zeros(self.embedding_dim, dtype=torch.float, device=self.device))
            else:
                index = self.index_map[_x]
                res.append(self.params[index // self.emb_step](torch.tensor(index % self.emb_step, device=self.device)))
        res = torch.cat(res).reshape(_shape + (self.embedding_dim,))
        return res

    def state_dict(self, *args, **kwargs):
        # self.register_parameter('map_keys', nn.Parameter(torch.tensor(['a', 'b'])))
        # for i in range(len(self.params)):
        #     self.add_module('params_{}'.format(i), self.params[i])
        res = super().state_dict(*args, **kwargs)
        # print('sate dict args:', args)
        prefix = args[1]
        res[prefix + 'map_keys'] = '\01'.join(self.index_map.keys())
        res[prefix + 'map_values'] = '\01'.join([str(v) for v in self.index_map.values()])
        return res


    def _load_from_state_dict(self, *args, **kwargs):
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs = args
        # print('state_dict meta data:', state_dict._metadata)
        # print('metadata:', getattr(state_dict, '_metadata', None))
        # print('_load_from_state_dict state_dict:', state_dict)
        # print('local meta:', local_metadata)
        # print(prefix, state_dict[prefix + 'map_keys'], state_dict[prefix + 'map_values'])
        if len(state_dict[prefix + 'map_keys']) > 0:
            for k, v in zip(state_dict[prefix + 'map_keys'].split('\01'), state_dict[prefix + 'map_values'].split('\01')):
                self.index_map[k] = int(v)

        index = 0
        while 1:
            key = prefix + 'params_{}'.format(index) + '.emb.weight'
            if key not in state_dict:
                break
            self.add_param()
            self.params[-1]._load_from_state_dict(
                state_dict,
                prefix + 'params_{}.'.format(index),
                local_metadata.get(prefix[:-1], {}),
                *args[3:]
            )
            index += 1
            # del state_dict[prefix + 'params_{}'.format(index) + '.weight']
            # del state_dict[prefix + 'params_{}'.format(index) + '.bias']
        del state_dict[prefix + 'map_keys']
        del state_dict[prefix + 'map_values']
        super()._load_from_state_dict(*args, **kwargs)
