import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, out_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.block_entry = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1x1 = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=strides) if use_1x1conv else None
        self.block_exit = nn.ReLU()

    def forward(self, x):
        y = self.block_entry(x)
        if self.conv1x1:
            x = self.conv1x1(x)
        y+=x
        return self.block_exit(y)

class ResNetBlock(nn.Module):
    def __init__(self, input_channels, out_channels, num_residuals, first_block=False):
        super(ResNetBlock, self).__init__()
        self.net_blocks = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.net_blocks.add_module(f'block_{i}', ResidualBlock(input_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                self.net_blocks.add_module(f'block_{i}', ResidualBlock(out_channels, out_channels))

    def forward(self, x):
        return self.net_blocks(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, name='18', **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.block_config = {
            '18': [2, 2, 2, 2],
            '34': [3, 4, 6, 3],
            '50': [3, 4, 6, 3],
            '101': [3, 4, 23, 3],
            '152': [3, 8, 36, 3]
        }
        assert name in self.block_config.keys(), f'name should be in {self.block_config.keys()}'
        self.name = name
        self.net_entry = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        in_channels = 64
        self.net_blocks = []
        for i, num_layers in enumerate(self.block_config[name]):
            out_channels = 64 * (2**i)
            self.net_blocks.append(ResNetBlock(in_channels, out_channels, num_layers, first_block=(i == 0)))
            in_channels = out_channels

        self.net_exit = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        self.net = nn.Sequential(self.net_entry, *self.net_blocks, self.net_exit)

    def forward(self, x):
        return self.net(x)

    def _get_name(self):
        return self.name