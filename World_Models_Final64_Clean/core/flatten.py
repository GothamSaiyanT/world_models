import torch 
from core.module import Module

class Flatten(Module):

    def __init__(self):
        super().__init__()
        #store the original shape for backward pass
        self.input_shape = None

    def forward(self,x):
        #input :(batch,channels,height,width)
        #(batch,channels*  heihts * width)

        self.input_shape = x.shape
        batch_size = x.shape[0]

        return x.reshape(batch_size,-1)

    def backward(self,grad_output):
        #restore the original tensor shape

        return grad_output.reshape(self.input_shape)

    def parameters(self):
        return []
    
    def children(self):
        return []