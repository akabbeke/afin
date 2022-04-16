import torch

from torch import nn

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = nn.Linear(n_inputs, 1)
        self.activation = nn.Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X


class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputSize, outputSize),
            nn.ReLU(),
            nn.Linear(inputSize, outputSize),
            nn.ReLU(),
            nn.Linear(inputSize, outputSize)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

