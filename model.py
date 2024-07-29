import torch
import torch.nn as nn

class SchedulingNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(SchedulingNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.network = nn.ModuleList()
        self.network.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.network.append(nn.ReLU())
        for i in range(5):
            self.network.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x

