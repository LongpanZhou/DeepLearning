from torch import nn

class MobileNet_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(MobileNet_Conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,stride,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

class MobileNet_Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(MobileNet_Block,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,stride,1,groups=in_channels,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)

class MobileNet_v1(nn.Module):
    def __init__(self, in_channels, classes):
        super(MobileNet_v1,self).__init__()
        self.features = nn.Sequential(
            MobileNet_Conv(in_channels,32,2),
            MobileNet_Block(32,64,1),
            MobileNet_Block(64,128,2),
            MobileNet_Block(128,128,1),
            MobileNet_Block(128,256,2),
            MobileNet_Block(256,256,1),
            MobileNet_Block(256,512,2),

            # 5 times
            MobileNet_Block(512,512,1),
            MobileNet_Block(512,512,1),
            MobileNet_Block(512,512,1),
            MobileNet_Block(512,512,1),
            MobileNet_Block(512,512,1),

            MobileNet_Block(512,1024,2),
            MobileNet_Block(1024,1024,1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,classes)
        )

    def forward(self,x):
        return self.classifier(self.features(x))