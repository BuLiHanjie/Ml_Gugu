import torch
from torch import nn

from torch_model.mlp import Mlp, Linear
from torch_model.torch_args.torch_param import DeepFmParam


class DeepFm(nn.Module):
    def __init__(self, param: DeepFmParam):
        super().__init__()
        self.deep_mlp = Mlp(
            in_features=param.deep_dim,
            hidden_layers=param.mlp_hidden_layers,
            activations=param.mlp_activations,
        )
        self.fm_linear = Linear(
            in_features=param.emedding_dim * 2 + param.mlp_hidden_layers[-1],
            out_features=param.linear_out_features,
            activation=param.linear_activaions
        )

    def forward(self, deep_term, first_order_emb, second_order_emb):
        '''
        deepfm
        :param deep_emb: None * F * K
        :param first_order_emb: None * F or [None] * F
        :param second_order_emb: None * F * K or [None * K] * F
        :return:
        '''
        # ---------- first order term ---------------
        # None * K
        if isinstance(first_order_emb, list):
            first_order_emb = torch.stack(first_order_emb, 1)
        first_order = first_order_emb
        # ---------- second order term ---------------
        # None * K
        if isinstance(second_order_emb, list):
            second_order_emb = torch.stack(second_order_emb, 1)
        # sum_square part
        second_order_emb_sum = torch.sum(second_order_emb, 1)
        second_order_emb_sum_square = second_order_emb_sum ** 2
        # square_sum part
        second_order_emb_square = second_order_emb ** 2
        second_order_emb_square_sum = torch.sum(second_order_emb_square, 1)
        # second order
        second_order = 0.5 * (second_order_emb_sum_square - second_order_emb_square_sum)

        # ---------- Deep component ----------
        # None * mlp_size
        deep_term = deep_term.reshape(deep_term.shape[0], -1)
        deep_dense = self.deep_mlp(deep_term)

        # ---------- DeepFM ----------
        # None * linaer_size
        concat_input = torch.cat([deep_dense, first_order, second_order], 1)
        res = self.fm_linear(concat_input)
        return res
