#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
import math
import torch
from torch import nn
import numpy as np
from torch.nn import functional
class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):  # 16
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()  # sigmoid
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SPAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=4):  # 16
        super(SPAttentionLayer, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1, padding=1,
                      dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.midd = nn.Sequential(
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel // reduction, kernel_size=3, stride=1,padding=1,
                      dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Conv2d(in_channels=channel // reduction, out_channels=1, kernel_size=3, padding=1,stride=1, dilation=1,
                              bias=False)

    def forward(self, x):
        x1 = self.head(x)
        SA = self.midd(x1)
        SA = self.tail(SA)
        return x * F.sigmoid(SA)

class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x.detach(), self.blur.detach()))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# if __name__ == '__main__':
#     import torch
#
#     for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
#
#
#         img = torch.zeros(2, 3, 20, 20)
#         net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
#         out = net(img)
#         print(out.size())


