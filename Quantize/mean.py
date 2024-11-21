import torch
from torch import nn
import torch.nn.functional as F


def quantize(x, num_bits=8):
    min_val = x.min()
    max_val = x.max()

    scale = (max_val - min_val) / (num_bits**2 - 1)

    q = torch.round((x - min_val) / scale).clamp(0, 2**num_bits - 1)

    return q.to(torch.uint8), scale, min_val


def dequantize(q, scale, min_val):
    return q * scale + min_val
