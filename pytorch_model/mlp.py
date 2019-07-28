import torch
import torch.nn as nn

from torch_model.str_to_function import str_to_activation


class Linear(nn.Module):

    def __init__(self, in_features, out_features, activation, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activate = str_to_activation(activation) if isinstance(activation, str) else activation

    def forward(self, x):
        x = self.linear(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_layers, activations, bias=True):
        super().__init__()
        if not isinstance(activations, list):
            activations = [activations] * len(hidden_layers)
        layers = list()
        for i, (_h, _activate) in enumerate(zip(hidden_layers, activations)):
            layers.append(
                Linear(
                    in_features=in_features if i == 0 else hidden_layers[i - 1],
                    out_features=_h,
                    activation=_activate,
                    bias=bias
                )
            )
        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.model(x)
