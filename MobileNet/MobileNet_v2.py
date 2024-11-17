from torch import nn

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(InvertedResidual,self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels*expansion),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels*expansion, in_channels*expansion, kernel_size=3, stride=stride, padding=1, groups=in_channels*expansion, bias=False),
            nn.BatchNorm2d(in_channels*expansion),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels*expansion, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNet_v2(nn.Module):
    def __init__(self, in_channels, classes):
        super(MobileNet_v2,self).__init__()

        self.configs = [
            # t, c, n, s
            [1,16,1,1],
            [6,24,2,2],
            [6,32,3,2],
            [6,64,4,2],
            [6,96,3,1],
            [6,160,3,2],
            [6,320,1,1]
        ]

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.blocks = []
        in_channels = 32
        for expansion, out_channels, num_blocks, stride in self.configs:
            for i in range(num_blocks):
                self.blocks.append(InvertedResidual(in_channels, out_channels, stride if i == 0 else 1, expansion))
                in_channels = out_channels

        self.blocks = nn.Sequential(*self.blocks)
        self.exit = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1280, classes)
        )

    def forward(self,x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)
        x = self.classifier(x)
        return x