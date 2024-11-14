import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.net_blocks = nn.Sequential()
        for i in range(num_convs):
            self.net_blocks.add_module(f'block_{i+1}', ConvBlock(in_channels + i * out_channels, out_channels))

    def forward(self, x):
        for blk in self.net_blocks:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        return x

class DenseNet(nn.Module):
    def __init__(self, in_channels, num_classes, name='121', **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.block_config = {'121': [6, 12, 24, 16],
                             '169': [6, 12, 32, 32],
                             '201': [6, 12, 48, 32],
                             '264': [6, 12, 64, 48]}
        assert name in self.block_config.keys(), f'name should be in {self.block_config.keys()}'
        self.name = name
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channels, growth_rate = 64, 32
        self.net_blocks = []
        for i, num_layers in enumerate(self.block_config[name]):
            self.net_blocks.append(DenseBlock(num_layers, num_channels, growth_rate))
            num_channels += num_layers * growth_rate
            if i != len(self.block_config[name]) - 1:
                self.net_blocks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.net = nn.Sequential(
            self.b1, *self.net_blocks,
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def _get_name(self):
        return self.name