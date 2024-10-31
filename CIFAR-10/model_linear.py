import torch.nn as nn

input_size = 3072
hidden_sizes = [768, 192, 48]
output_size = 10

class NN_Linear(nn.Module):
    def __init__(self):
        super(NN_Linear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1],hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x = x.view(x.size(0), -1)
        return self.model(x)