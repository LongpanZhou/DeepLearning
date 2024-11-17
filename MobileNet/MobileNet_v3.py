from torch import nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InvertedResidual(nn.Module):
    def __init__(self, kernel_size, expansion, in_channels, out_channels, stride, SE=False, NL='RE'):
        super(InvertedResidual, self).__init__()
        padding = (kernel_size - 1)//2
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(expansion),
            nn.ReLU() if NL=='RE' else HSwish(),

            nn.Conv2d(expansion,expansion,kernel_size,stride,padding, groups=expansion, bias=False),
            nn.BatchNorm2d(expansion),

            SE_Block(expansion) if SE else Identity(expansion),
            nn.ReLU() if NL=='RE' else HSwish(),

            nn.Conv2d(expansion,out_channels,1,1,0,bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNet_v3(nn.Module):
    def __init__(self, in_channels, classes, name="small"):
        super(MobileNet_v3,self).__init__()
        self.block_config = {
            'large': [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ],

            'small': [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        }
        assert name in self.block_config.keys(), f'name should be in {self.block_config.keys()}'
        self.name = name

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish(inplace=True)
        )

        in_channels = 16
        self.net_blocks = nn.Sequential()
        for i, (k, exp, c, se, nl, s) in enumerate(self.block_config[name]):
            self.net_blocks.add_module(f'block{i}', InvertedResidual(k, exp, in_channels, c, s, se, nl))
            in_channels = c
        exit_channels = in_channels

        self.exit = nn.Sequential(
            nn.Conv2d(exit_channels, 576 if name == "small" else 960, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576 if name == "small" else 960),
            HSwish(inplace=True),

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(576 if name == "small" else 960, 1024 if name == "small" else 1280, kernel_size=1, stride=1, padding=0, bias=False),
            HSwish(inplace=True),

            nn.Conv2d(1024 if name == "small" else 1280, classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.net_blocks(x)
        x = self.exit(x)
        return x.flatten(1)

    def _get_name(self):
        return self.name