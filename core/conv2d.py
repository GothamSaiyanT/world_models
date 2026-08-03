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
        """
        Vectorised convolution using unfold + einsum instead of a
        triple-nested Python loop over batch/height/width. This is
        the same mathematical operation as before, just implemented
        without iterating pixel-by-pixel in Python, which was the
        main source of the extreme slowness during training.
        """

        self.input = x

        batch_size, channels, height, width = x.shape

        k = self.kernel_size
        s = self.stride

        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding)
            )
            height = height + 2 * self.padding
            width = width + 2 * self.padding

        out_height = (
            height - k
        ) // s + 1

        out_width = (
            width - k
        ) // s + 1

        weights = self.weight.data
        bias = self.bias.data

        # Extract all k x k patches at once: (batch, channels*k*k, num_patches)
        patches = torch.nn.functional.unfold(
            x,
            kernel_size=k,
            stride=s
        )

        # Flatten the weight tensor to (out_channels, channels*k*k)
        weight_matrix = weights.view(self.out_channels, -1)

        # (out_channels, channels*k*k) x (batch, channels*k*k, num_patches)
        # -> (batch, out_channels, num_patches)
        output = torch.einsum(
            "oc,bcl->bol",
            weight_matrix,
            patches
        )

        output = output + bias.view(1, -1, 1)

        output = output.view(
            batch_size,
            self.out_channels,
            out_height,
            out_width
        )

        return output

    def parameters(self):

        return [
            self.weight,
            self.bias
        ]