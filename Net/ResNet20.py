import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
  expansion = 1

  def __init__(self, input_channels, num_channels,
               use_1x1conv=False, strides=1):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, num_channels,
                           kernel_size=3, padding=1, stride=strides)
    self.conv2 = nn.Conv2d(num_channels, num_channels,
                           kernel_size=3, padding=1)
    if use_1x1conv:
      self.conv3 = nn.Conv2d(input_channels, num_channels,
                             kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)

  def forward(self, X):
    Y = F.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))
    if self.conv3:
      X = self.conv3(X)
    Y += X
    return F.relu(Y)


class ResNet20(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super().__init__()
    self.in_channels = 16
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True))
    self.conv2 = self._make_layer(block, 16, num_blocks[0], 1)
    self.conv3 = self._make_layer(block, 32, num_blocks[1], 2)
    self.conv4 = self._make_layer(block, 64, num_blocks[2], 2)
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64 * block.expansion, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels, out_channels, stride))
      self.in_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.avg_pool(out)
    out = out.view(out.size(0), -1)
    return self.fc(out)


def make_ResNet20(num_classes=10, **kargs):
  return ResNet20(Residual, [3, 3, 3], num_classes=num_classes)
