"""Networks for the image-classification tables of X-SAM, Eigen-SAM and WSAM.

None of the papers gives its exact definitions, so these are the usual CIFAR versions (ours):

    resnet18, resnet50  He et al. (2016) with a 3x3 stride-1 stem and no max-pool, the common
                        CIFAR adaptation (github.com/kuangliu/pytorch-cifar)
    wrn28_10            Zagoruyko & Komodakis (2016), pre-activation, widen factor 10, no dropout
    alexnet             Krizhevsky et al. (2012) conv stack 64-192-384-256-256 sized for 32x32
                        inputs; ceil-mode pooling so 28x28 Fashion-MNIST also fits
    vit_b_16            torchvision's ImageNet-1k checkpoint (Eigen-SAM Sec. 6.3), new zero head

All take `in_channels` (1 for Fashion-MNIST) and `num_classes`.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

TORCH_HOME = os.path.join(os.environ.get("DATA_ROOT", "data"), "torch")  # pretrained weights cache


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, c, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != c:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, c, 1, stride, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, c, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, 4 * c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * c)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != 4 * c:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, 4 * c, 1, stride, bias=False), nn.BatchNorm2d(4 * c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(self.bn3(self.conv3(out)) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        stages, cin = [], 64
        for i, n in enumerate(layers):
            c = 64 * 2 ** i
            for j in range(n):
                stages.append(block(cin, c, 2 if i > 0 and j == 0 else 1))
                cin = c * block.expansion
        self.layers = nn.Sequential(*stages)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        out = self.layers(F.relu(self.bn1(self.conv1(x))))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


class WideBlock(nn.Module):
    def __init__(self, cin, c, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.shortcut = None if stride == 1 and cin == c else nn.Conv2d(cin, c, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(self.conv1(o))))
        return out + (x if self.shortcut is None else self.shortcut(o))


class WideResNet(nn.Module):
    def __init__(self, depth, width, in_channels, num_classes):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16 * width, 32 * width, 64 * width]
        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False)
        blocks, cin = [], widths[0]
        for i, c in enumerate(widths[1:]):
            for j in range(n):
                blocks.append(WideBlock(cin, c, 2 if i > 0 and j == 0 else 1))
                cin = c
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = F.relu(self.bn(self.blocks(self.conv1(x))))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


def alexnet(in_channels, num_classes):
    pool = lambda: nn.MaxPool2d(2, ceil_mode=True)
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 11, 4, 5), nn.ReLU(inplace=True), pool(),
        nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True), pool(),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), pool(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))


def vit_b_16(num_classes):
    from torchvision.models import ViT_B_16_Weights, vit_b_16 as tv_vit
    torch.hub.set_dir(TORCH_HOME)
    model = tv_vit(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    nn.init.zeros_(model.heads.head.weight)
    nn.init.zeros_(model.heads.head.bias)
    return model


def build_model(name, in_channels, num_classes):
    if name == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)
    if name == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], in_channels, num_classes)
    if name == "wrn28_10":
        return WideResNet(28, 10, in_channels, num_classes)
    if name == "alexnet":
        return alexnet(in_channels, num_classes)
    if name == "vit_b_16":
        return vit_b_16(num_classes)
    raise ValueError(name)


MODELS = ["resnet18", "resnet50", "wrn28_10", "alexnet", "vit_b_16"]
