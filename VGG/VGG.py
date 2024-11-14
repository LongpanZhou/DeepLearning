import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, in_channels, classes, name="A", **kwargs):
        super(VGG, self).__init__(**kwargs)
        self.block_config  = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }
        assert name in self.block_config , f"Block configuration {name} not supported"
        self.name = name
        self.features = self.VGG_block(in_channels, self.block_config [name], batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, classes)
        )

    def VGG_block(self, in_channels, cfg, batch_norm=False):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _get_name(self):
        return self.name