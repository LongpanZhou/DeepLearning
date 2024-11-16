import torch
import torch.nn as nn
from SE_Block import SE_Block

# We need to go deeper
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SE_Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(SE_Inception, self).__init__()
        # Path 1
        self.p1 = BasicConv2d(in_channels, c1, kernel_size=1, padding=0)

        # Path 2
        self.p2 = nn.Sequential(
            BasicConv2d(in_channels, c2[0], kernel_size=1, padding=0),
            BasicConv2d(c2[0], c2[1], kernel_size=3, padding=1)
        )

        # Path 3
        self.p3 = nn.Sequential(
            BasicConv2d(in_channels, c3[0], kernel_size=1, padding=0),
            BasicConv2d(c3[0], c3[1], kernel_size=5, padding=2)
        )

        # Path 4
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, c4, kernel_size=1, padding=0)
        )

        # SE
        self.SE = SE_Block(c1 + c2[1] + c3[1] + c4)

    def forward(self, x):
        x = torch.cat((self.p1(x), self.p2(x), self.p3(x), self.p4(x)), dim=1)
        return self.SE(x)

class SE_GoogLeNet(nn.Module):
    def __init__(self, in_channels, classes, **kwargs):
        super(SE_GoogLeNet, self).__init__(**kwargs)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            SE_Inception(192, 64, (96, 128), (16, 32), 32),
            SE_Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            SE_Inception(480, 192, (96, 208), (16, 48), 64),
            SE_Inception(512, 160, (112, 224), (24, 64), 64),
            SE_Inception(512, 128, (128, 256), (24, 64), 64),
            SE_Inception(512, 112, (144, 288), (32, 64), 64),
            SE_Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            SE_Inception(832, 256, (160, 320), (32, 128), 128),
            SE_Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(),
            nn.Flatten()
        )

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, nn.Linear(1024, classes))

    def forward(self, x):
        return self.net(x)