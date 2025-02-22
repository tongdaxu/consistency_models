import pathlib
import cv2
import numpy as np
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def transposed_conv(in_channels, out_channels, stride=2):
    ''' transposed conv with same padding '''
    kernel_size, padding = {
        2: (4, 1),
        4: (8, 2),
        16: (32, 8),
    }[stride]
    layer = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False)
    return layer


class PlanarSegHead(nn.Module):

    def __init__(self, bottleneck_channels, in_features=2048, num_classes=5):
        super().__init__()
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm2d(in_features)
        self.fc_conv = nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, bias=False)

        self.clf1 = nn.Conv2d(in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.clf2 = nn.Conv2d(in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.clf3 = nn.Conv2d(in_features // 2, bottleneck_channels, kernel_size=1, stride=1, bias=False)

        self.dec1 = transposed_conv(bottleneck_channels, bottleneck_channels, stride=2)
        self.dec2 = transposed_conv(bottleneck_channels, bottleneck_channels, stride=2)
        self.dec3 = transposed_conv(bottleneck_channels, bottleneck_channels, stride=16)

        self.fc_stage2 = nn.Conv2d(bottleneck_channels, num_classes, kernel_size=1, stride=1, bias=False)

        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *feats):
        e7, e6, e5 = feats

        x = self.drop1(e7)
        x = self.fc_conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop2(x)

        c = self.clf1(x)           # 5 x 5 x 5
        d6 = self.dec1(c)          # 5 x 10 x 10

        d6_b = self.clf2(e6)       # 5 x 10 x 10
        d5 = self.dec2(d6_b + d6)  # 5 x 20 x 20

        d5_b = self.clf3(e5)       # 5 x 20 x 20
        d0 = self.dec3(d5_b + d5)  # 5 x 320 x 320

        d = self.fc_stage2(d0)
        return d


class ResPlanarSeg(nn.Module):

    def __init__(self, pretrained=True, backbone='resnet101'):
        super().__init__()
        BackBone = getattr(models, backbone)
        self.resnet = BackBone(pretrained=pretrained)
        self.planar_seg = PlanarSegHead(bottleneck_channels=37, in_features=self.resnet.fc.in_features)

    def forward(self, x):
        '''
        x: 3 x 320 x 320
        '''
        x = self.resnet.conv1(x)       # 64 x 160 x 160
        x = self.resnet.bn1(x)
        e1 = self.resnet.relu(x)
        e2 = self.resnet.maxpool(e1)   # 64 x 80 x 80
        e3 = self.resnet.layer1(e2)    # 256 x 80 x 80
        e4 = self.resnet.layer2(e3)    # 512 x 40 x 40
        e5 = self.resnet.layer3(e4)    # 1024  x 20 x 20
        e6 = self.resnet.layer4(e5)    # 2048 x 10 x 10
        e7 = self.resnet.maxpool(e6)   # 2048 x 5 x 5

        return self.planar_seg(e7, e6, e5)

class LayoutSeg(pl.LightningModule):

    def __init__(
        self, lr: float = 1e-4, backbone: str = 'resnet101',
        l1_factor: float = 0.2, l2_factor: float = 0.0, edge_factor: float = 0.2
    ):
        super().__init__()
        self.lr = lr
        self.model = ResPlanarSeg(pretrained=True, backbone=backbone)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.edge_factor = edge_factor
        self.save_hyperparameters()

    def forward(self, inputs):
        scores = self.model(inputs)
        _, outputs = torch.max(scores, 1)
        return scores, outputs

def label_as_rgb_visual(x):
    """ Make segment tensor into colorful image
    Args:
        x (torch.Tensor): shape in (N, H, W) or (N, 1, H, W)
        colors (tuple or list): list of RGB colors, range from 0 to 1.
    Returns:
        canvas (torch.Tensor): colorized tensor in the shape of (N, C, H, W)
    """
    colors = [
        [0.9764706, 0.27058825, 0.3647059], [1., 0.8980392, 0.6666667],
        [0.5647059, 0.80784315, 0.70980394], [0.31764707, 0.31764707, 0.46666667],
        [0.94509804, 0.96862745, 0.8235294]]

    if x.dim() == 4:
        x = x.squeeze(1)
    assert x.dim() == 3

    n, h, w = x.size()
    palette = torch.tensor(colors).to(x.device)
    labels = torch.arange(x.max() + 1).to(x)

    canvas = torch.zeros(n, h, w, 3).to(x.device)
    for color, lbl_id in zip(palette, labels):
        if canvas[x == lbl_id].size(0):
            canvas[x == lbl_id] = color

    return canvas.permute(0, 3, 1, 2)
