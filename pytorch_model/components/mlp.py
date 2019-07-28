import torch
import torch.nn as nn

# from torch_model.str_to_function import str_to_activation
from torch_model.torch_args.torch_param import LinearParam, MlpParam


class Linear(nn.Module):

    def __init__(self, param: LinearParam):
        super().__init__()
        self.linear = nn.Linear(
            param.in_features,
            param.out_features,
            bias=param.bias
        )
        self.activate = param.activation

    def forward(self, x):
        x = self.linear(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class Mlp(nn.Module):

    def __init__(self, param: MlpParam):
        super().__init__()
        layers = list()
        for i, (_h, _activate, _b) in enumerate(zip(param.hidden_layers, param.activations, param.bias)):
            layers.append(
                Linear(
                    LinearParam(
                        in_features=param.in_features,
                        out_features=_h,
                        activation=_activate,
                        bias=_b
                    )
                )
            )
        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.model(x)

