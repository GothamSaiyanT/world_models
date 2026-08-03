import torch

from core.parameter import Parameter
from core.module import Module


class Conv2D(Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = (
            in_channels *
            kernel_size *
            kernel_size
        )

        std = (2.0 / fan_in) ** 0.5

        self.weight = Parameter(

            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ) * std

        )

        self.bias = Parameter(

            torch.zeros(out_channels)

        )

    def forward(self, x):

        self.input = x

        batch_size, channels, height, width = x.shape

        k = self.kernel_size
        s = self.stride

        out_height = (
            height - k
        ) // s + 1

        out_width = (
            width - k
        ) // s + 1

        weights = self.weight.data
        bias = self.bias.data

        output = torch.empty(
            batch_size,
            self.out_channels,
            out_height,
            out_width,
            device=x.device,
            dtype=x.dtype
        )

        for b in range(batch_size):

            for i in range(out_height):

                row = i * s

                for j in range(out_width):

                    col = j * s

                    patch = x[
                        b,
                        :,
                        row:row + k,
                        col:col + k
                    ]

                    values = (
                        weights * patch
                    ).sum(dim=(1, 2, 3))

                    values += bias

                    output[
                        b,
                        :,
                        i,
                        j
                    ] = values

        return output

    def parameters(self):

        return [
            self.weight,
            self.bias
        ]