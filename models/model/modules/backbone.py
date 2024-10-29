import torch
from torch import nn
import torchvision as tv
from .naive_init import naive_init_module


class ResNet34_Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34_Backbone, self).__init__()
        resnet = tv.models.resnet34(pretrained=pretrained)

        # Initial layers
        self.conv1 = resnet.conv1  # (B, 64, H/2, W/2)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # (B, 64, H/4, W/4)

        # ResNet layers
        self.layer1 = resnet.layer1  # (B, 64, H/4, W/4)
        self.layer2 = resnet.layer2  # (B, 128, H/8, W/8)
        self.layer3 = resnet.layer3  # (B, 256, H/16, W/16)
        self.layer4 = resnet.layer4  # (B, 512, H/32, W/32)

        # Initialize weights
        for m in self.modules():
            naive_init_module(m)

    def forward(self, x):
        skips = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # skips[0]: After maxpool (B, 64, H/4, W/4)
        skips.append(x)

        x = self.layer1(x)
        # skips[1]: After layer1 (B, 64, H/4, W/4)
        skips.append(x)

        x = self.layer2(x)
        # skips[2]: After layer2 (B, 128, H/8, W/8)
        skips.append(x)

        x = self.layer3(x)
        # skips[3]: After layer3 (B, 256, H/16, W/16)
        skips.append(x)

        x = self.layer4(x)
        # skips[4]: After layer4 (B, 512, H/32, W/32)
        skips.append(x)

        return skips  # List of feature maps from different stages

class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)