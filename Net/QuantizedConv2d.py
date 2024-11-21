import torch
import torch.nn as nn
import torch.nn.functional as F

import Quantize
import Quantize.mean


class QuantizedConv2d(nn.Module):
    def __init__(self, conv_layer, num_bits=8):
        super(QuantizedConv2d, self).__init__()  # 拷贝参数
        # self.weight = conv_layer.weigth.data
        self.bias = conv_layer.bias.data if conv_layer.bias is not None else None
        self.q_weight, self.w_scale, self.w_min = Quantize.mean.quantize(
            conv_layer.weight.data, num_bits=num_bits
        )
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.num_bits = num_bits

    def forward(self, x):
        return F.conv2d(
            x,
            Quantize.mean.dequantize(self.q_weight, self.w_scale, self.w_min),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def copy_from_full_precision(self, full_pre_model):
        self.bias = (
            full_pre_model.bias.data if full_pre_model.bias is not None else None
        )
        self.q_weight, self.w_scale, self.w_min = Quantize.mean.quantize(
            full_pre_model.weight.data, num_bits=self.num_bits
        )
        self.stride = full_pre_model.stride
        self.padding = full_pre_model.padding
        self.dilation = full_pre_model.dilation
        self.groups = full_pre_model.groups

    def __repr__(self):
        return f"QuantizedConv2d({self.q_weight.shape[0]}, {self.q_weight.shape[1]}, kernel_size=({self.q_weight.shape[2]}, {self.q_weight.shape[3]}), stride={self.stride}, padding={self.padding})"
