import torch.nn as nn

class Network(nn.Module):

    def __init__(self, input_dim):
        super(Network, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = 50

        self.linear_one = nn.Linear(self.input_dim, self.hidden_size)
        self.relu_one = nn.ReLU()
        self.linear_two = nn.Linear(self.hidden_size, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_one(x)
        x = self.relu_one(x)
        x = self.linear_two(x)
        return self.out(x)