import torch

from core.parameter import Parameter
from core.module import Module

class Conv2D(Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride =1,
            padding = 0
    ):
        super().__init__()
        self.in_channels = in_channels

        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        #He initialization
        fan_in = in_channels * kernel_size * kernel_size

        std = (2.0 / fan_in)** 0.5

        weights = torch.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * std

        self.weight = Parameter(weights)
        self.bias  =Parameter(torch.zeros(out_channels))

    def forward(self,x):
        #we need to save the original for a case where we are doing backpropagation
        self.input = x

        batch_size,channels,height,width = x.shape
        out_height = (height - self.kernel_size + 2 * self.padding)// self.stride + 1
        out_width = (width - self.kernel_size + 2*self.padding)//self.stride + 1

        output = torch.zeros(
            batch_size,
            self.out_channels,
            out_height,
            out_width
        )

        for b in range(batch_size):
            for f in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        
                        row_start = i * self.stride
                        col_start = j * self.stride

                        patch = x[
                            b,
                            :,
                            row_start:row_start + self.kernel_size,
                            col_start:col_start + self.kernel_size
                        ]

                        kernel = self.weight.data[f]

                        value = torch.sum(
                            patch * kernel
                        )

                        value +=self.bias.data[f]

                        output[b,f,i,j] = value


        return output

    def parameters(self):
        return [self.weight,self.bias]
