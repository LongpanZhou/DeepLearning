import torch
from torch import nn

# Not sure if I implemented the block correctly
class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
        self.stride = stride
        self.bottle_neck = out_channels // 2

        self.branch_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, self.bottle_neck, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.bottle_neck),
            nn.ReLU(inplace=True)
        )

        self.branch_main = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else self.bottle_neck, self.bottle_neck, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.bottle_neck),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.bottle_neck, self.bottle_neck, kernel_size=3, stride=stride, padding=1, groups=self.bottle_neck, bias=False),
            nn.BatchNorm2d(self.bottle_neck),

            nn.Conv2d(self.bottle_neck, self.bottle_neck, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.bottle_neck),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x_proj, x_main = torch.chunk(x, 2, dim=1)
            out = torch.cat((x_proj, self.branch_main(x_main)), dim=1)
        else:
            x_proj = x
            x_main = x
            out = torch.cat((self.branch_proj(x_proj), self.branch_main(x_main)), dim=1)

        return self.channel_shuffle(out)

    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.groups == 0, f"Number of channels {num_channels} must be divisible by number of groups {self.groups}"
        channels_per_group = num_channels // self.groups

        x = x.reshape(batch_size, self.groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, num_channels, height, width)
        return x

class ShuffleNet_v2(nn.Module):
    def __init__(self, in_channels, classes, model_size='1.0x', group=3):
        super(ShuffleNet_v2, self).__init__()
        self.groups = {
            1: [144, 288, 576],
            2: [200, 400, 800],
            3: [240, 480, 960],
            4: [272, 544, 1088],
            8: [384, 768, 1536]
        }
        assert group in self.groups.keys(), f"Group value must be in {self.groups.keys()}"

        self.model_sizes = {'0.5x': 0.5, '1.0x': 1.0, '1.5x': 1.5, '2.0x': 2.0}
        assert model_size in self.model_sizes.keys(), f"Model size must be in {self.model_sizes.keys()}"

        self.stage_out_channels = [int(c * self.model_sizes[model_size]) for c in self.groups[group]]
        self.model_size = model_size

        # Entry Block
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Stage 2
        self.stage2 = nn.Sequential(
            ShuffleBlock(24, self.stage_out_channels[0], group, stride=2),
            *[ShuffleBlock(self.stage_out_channels[0], self.stage_out_channels[0], group, stride=1) for _ in range(3)]
        )

        #Stage 3
        self.stage3 = nn.Sequential(
            ShuffleBlock(self.stage_out_channels[0], self.stage_out_channels[1], group, stride=2),
            *[ShuffleBlock(self.stage_out_channels[1], self.stage_out_channels[1], group, stride=1) for _ in range(7)]
        )

        #Stage 4
        self.stage4 = nn.Sequential(
            ShuffleBlock(self.stage_out_channels[1], self.stage_out_channels[2], group, stride=2),
            *[ShuffleBlock(self.stage_out_channels[2], self.stage_out_channels[2], group, stride=1) for _ in range(3)]
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.stage_out_channels[-1], classes)
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x