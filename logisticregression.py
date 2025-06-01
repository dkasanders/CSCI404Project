import torch
import sys
import numpy as np
from torch import nn

class LogisticRegression(torch.nn.Module):
    def __init__(self, C, D):

        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(C, D)

    def forward(self, x):
       return self.linear(x)

    

