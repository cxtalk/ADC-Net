#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from opts import *

def init_weights(modules):
    pass
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

class ConvDirec(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super(ConvDirec,self).__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad, dilation=dilation)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class MSRB(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(MSRB, self).__init__()

        self.x1_1 = ConvDirec(in_channels, out_channels,3,1)
        self.x2_1 = ConvDirec(in_channels, out_channels,5,1)
        self.x1_2 = ConvDirec(in_channels, out_channels,3,1)
        self.x2_2 = ConvDirec(in_channels, out_channels,5,1)

        self.out1 = nn.Conv2d(out_channels*2, out_channels, 1, padding=0)
        self.out2 = nn.Conv2d(out_channels, in_channels, 3, padding=1)

        self.se = SELayer(in_channels)
        self.sp = SPAttentionLayer(in_channels)

    def forward(self, x):
            x12_1 = self.x1_1(x) + self.x2_1(x)

            x3 = self.x1_2(x12_1)
            x5 = self.x2_2(x12_1)

            c1 = torch.cat([x3, x5], dim=1)

            c3 = self.out1(c1)
            c4 = self.out2(c3)

            c4 = self.sp(self.se(c4))
            out = x + c4

            return out


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()

        num=32

        self.entry = nn.Conv2d(3, num, 3, 1, 1)

        self.b1 = MSRB(num, num)
        self.b2 = MSRB(num, num)
        self.b3 = MSRB(num, num)
        self.b4 = MSRB(num, num)
        self.b5 = MSRB(num, num)
        self.b6 = MSRB(num, num)
        self.b7 = MSRB(num, num)
        self.b8 = MSRB(num, num)
        self.c1 = BasicBlock(num * 2, num, 1, 1, 0)
        self.c2 = BasicBlock(num * 3, num, 1, 1, 0)
        self.c3 = BasicBlock(num * 4, num, 1, 1, 0)
        self.c4 = BasicBlock(num * 5, num, 1, 1, 0)
        self.c5 = BasicBlock(num * 6, num, 1, 1, 0)
        self.c6 = BasicBlock(num * 7, num, 1, 1, 0)
        self.c7 = BasicBlock(num * 8, num, 1, 1, 0)
        self.c8 = BasicBlock(num * 9, num, 1, 1, 0)

        self.exit = nn.Conv2d(num, 3, 3, 1, 1)

    def forward(self, x):
        x0=self.entry(x)
        c0 = o0 = x0

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        b4 = self.b4(o3)
        c4 = torch.cat([c3, b4], dim=1)
        o4 = self.c4(c4)

        b5 = self.b5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)

        b7 = self.b7(o6)
        c7 = torch.cat([c6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)

        out = self.exit(o8)
        out = torch.tanh(out)
        derain= x + out

        return derain
if __name__ == '__main__':
        img = torch.zeros(2, 1, 20, 20)
        net = Net()
        model=net(img)
        print(model.size())
        params = list(net.parameters())
        k = 0
        for i in params:
            l = 1
            # print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
            # print("该层参数和：" + str(l))
            k = k + l
        print("总参数数量和：" + str(k))

        print('cliqueNet parameters:', sum(param.numel() for param in net.parameters()))
