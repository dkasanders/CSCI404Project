import torch
import argparse
import sys
import numpy as np
from torch import nn

"""
This class defines a deep neural network with the following properties

C: Output dimension
L: Number of layers, defined as a comma separated list of units by layers. For example "32x1,16x2" would have a hidden layer of 32 nodes, and then two hidden layers with 16 nodes.
D: Input dimension
activation_function: Non-linear activation function, either 'relu', 'tanh', or 'sigmoid'



"""

class DeepNeuralNet(torch.nn.Module):
    def __init__(self, C, L, D, activation_function):
        """
        In the constructor we instantiate our weights and assign them to
        member variables.
        """

        activation_functions = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }

       
        super(DeepNeuralNet, self).__init__()
        self.linears = nn.ModuleList()
        self.activation = activation_functions[activation_function]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        if(L == '0x0'):
            self.linears.append(nn.Linear(D, C))
        else:
            parts = L.split(',')
            prev_dimension = D
            for i in range(len(parts)):
                dimensions = parts[i].split('x')
                hidden_units = int(dimensions[0])
                self.linears.append(nn.Linear(prev_dimension, hidden_units))
                for j in range(int(dimensions[1])-1):
                    self.linears.append(nn.Linear(hidden_units, hidden_units))
                prev_dimension = hidden_units
            self.linears.append(nn.Linear(prev_dimension, C))

        for name, param in self.named_parameters():
            print(name,param.data.shape)

        
        

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. 
        """
        # y = softmax(wTx + b)

        

        for i in range(len(self.linears)):
            x = self.activation(self.linears[i](x))

        return x