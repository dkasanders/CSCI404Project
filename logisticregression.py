import torch
import sys
import numpy as np
from torch import nn

class LogisticRegression(torch.nn.Module):
    def __init__(self, C, D):

        super(LogisticRegression, self).__init__()
        self.linears = nn.ModuleList()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(x @ self.linears[0].weight.T + self.linears[0].bias)

        softmax = torch.nn.Softmax(dim=0)
        y_pred = softmax(x)
        return y_pred

    

