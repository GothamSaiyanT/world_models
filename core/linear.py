import torch 

from core.parameter import Parameter
from core.module import Module

class Linear(Module):

    def __init__(self,in_features,out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        #He - Inintialization

        std = (2.0 / in_features) ** 0.5

        weights = torch.randn(
            out_features,
            in_features
        ) * std

        self.weight = Parameter(weights)

        self.bias = Parameter(
            torch.zeros(out_features)
        )

    def forward(self,x):

        #self.input = x

        return x @ self.weight.data.T + self.bias.data

    def parameters(self):
        return [self.weight,self.bias]