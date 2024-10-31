import torch.nn as nn

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x = x.view(x.size(0), -1)
        return self.model(x)