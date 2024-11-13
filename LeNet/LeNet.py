import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self,in_channels,classes,**kwargs):
        super(LeNet5,self).__init__(**kwargs)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, classes)              #This layer is suppose to be Gaussian Activiation
        )

    def forward(self,x):
        return self.classifier(self.feature_extractor(x))