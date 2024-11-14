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

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.net_entry = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.net_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, use_1x1conv=True, strides=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, use_1x1conv=True, strides=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, use_1x1conv=True, strides=2),
            ResidualBlock(512, 512)
        )
        self.net_exit = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        self.net = nn.Sequential(self.net_entry, self.net_blocks, self.net_exit)

    def forward(self, x):
        return self.net(x)
