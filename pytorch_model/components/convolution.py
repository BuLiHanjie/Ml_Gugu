import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_same(_input, filter, stride, dilation):
    res_pad = list()
    if isinstance(filter, int):
        filter = [filter] * (len(_input.shape) - 2)
    if isinstance(stride, int):
        stride = [stride] * len(filter)
    if isinstance(dilation, int):
        dilation = [dilation] * len(filter)
    for input_rows, filter_rows, _stride, _dilation in zip(_input.shape[2:], filter, stride, dilation):
        effective_filter_size_rows = (filter_rows - 1) * _dilation + 1
        out_rows = (input_rows + _stride - 1) // _stride
        padding_needed = max(0, (out_rows - 1) * _stride + effective_filter_size_rows -
                             input_rows)
        padding_top = padding_needed // 2
        padding_bottom = padding_needed - padding_top
        res_pad = [padding_top, padding_bottom] + res_pad
    return F.pad(_input, res_pad, mode='constant', value=0)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.model = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def forward(self, x):
        if self.padding == 'same':
            x = conv_same(x, self.kernel_size, self.stride, self.dilation)
        return self.model(x)


class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.model = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def forward(self, x):
        if self.padding == 'same':
            x = conv_same(x, self.kernel_size, self.stride, self.dilation)
        return self.model(x)
