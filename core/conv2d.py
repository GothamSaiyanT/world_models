import torch

from core.parameter import Parameter
from core.module import Module


def zero_pad_2d(x, padding):
    """
    Adds zero padding around a 4D image tensor.

    Input shape:
        (batch_size, channels, height, width)

    No torch.nn or torch.nn.functional is used.
    """

    if padding == 0:
        return x

    if padding < 0:
        raise ValueError(
            "Padding cannot be negative."
        )

    batch_size, channels, height, width = x.shape

    # Add padding to the left and right
    left_padding = torch.zeros(
        batch_size,
        channels,
        height,
        padding,
        device=x.device,
        dtype=x.dtype
    )

    right_padding = torch.zeros(
        batch_size,
        channels,
        height,
        padding,
        device=x.device,
        dtype=x.dtype
    )

    horizontally_padded = torch.cat(
        (
            left_padding,
            x,
            right_padding
        ),
        dim=3
    )

    padded_width = width + (2 * padding)

    # Add padding to the top and bottom
    top_padding = torch.zeros(
        batch_size,
        channels,
        padding,
        padded_width,
        device=x.device,
        dtype=x.dtype
    )

    bottom_padding = torch.zeros(
        batch_size,
        channels,
        padding,
        padded_width,
        device=x.device,
        dtype=x.dtype
    )

    padded = torch.cat(
        (
            top_padding,
            horizontally_padded,
            bottom_padding
        ),
        dim=2
    )

    return padded


class Conv2D(Module):
    """
    Custom vectorised 2D convolution.

    This implementation does not use:

        torch.nn
        torch.nn.functional
        torch.nn.Conv2d

    It manually performs convolution by:

        1. Extracting sliding image patches.
        2. Multiplying patches by filter weights.
        3. Summing the products.
        4. Adding the bias.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0
    ):

        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than zero."
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be greater than zero."
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be greater than zero."
            )

        if stride <= 0:
            raise ValueError(
                "stride must be greater than zero."
            )

        if padding < 0:
            raise ValueError(
                "padding cannot be negative."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.input = None

        # He initialisation
        fan_in = (
            in_channels
            * kernel_size
            * kernel_size
        )

        standard_deviation = (
            2.0 / fan_in
        ) ** 0.5

        self.weight = Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ) * standard_deviation
        )

        self.bias = Parameter(
            torch.zeros(
                out_channels
            )
        )

    def forward(self, x):
        """
        Performs vectorised convolution.

        Input:
            x with shape:
            (batch_size, in_channels, height, width)

        Output:
            tensor with shape:
            (
                batch_size,
                out_channels,
                output_height,
                output_width
            )
        """

        if x.ndim != 4:
            raise ValueError(
                "Conv2D input must have four dimensions: "
                "(batch, channels, height, width)."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"but received {x.shape[1]}."
            )

        self.input = x

        # Apply custom zero padding
        x = zero_pad_2d(
            x,
            self.padding
        )

        _, _, padded_height, padded_width = x.shape

        kernel_size = self.kernel_size
        stride = self.stride

        if kernel_size > padded_height:
            raise ValueError(
                "Kernel height is larger than the input height."
            )

        if kernel_size > padded_width:
            raise ValueError(
                "Kernel width is larger than the input width."
            )

        output_height = (
            padded_height - kernel_size
        ) // stride + 1

        output_width = (
            padded_width - kernel_size
        ) // stride + 1

        # Extract all sliding patches simultaneously.
        #
        # Resulting shape:
        #
        # (
        #     batch_size,
        #     in_channels,
        #     output_height,
        #     output_width,
        #     kernel_size,
        #     kernel_size
        # )
        patches = (
            x.unfold(
                dimension=2,
                size=kernel_size,
                step=stride
            )
            .unfold(
                dimension=3,
                size=kernel_size,
                step=stride
            )
        )

        # Multiply every image patch by every convolution filter.
        #
        # b = batch
        # c = input channel
        # h = output row
        # w = output column
        # i = kernel row
        # j = kernel column
        # o = output channel
        #
        # patches:
        #     (b, c, h, w, i, j)
        #
        # weights:
        #     (o, c, i, j)
        #
        # output:
        #     (b, o, h, w)
        output = torch.einsum(
            "bchwij,ocij->bohw",
            patches,
            self.weight.data
        )

        # Add one bias value per output channel
        output = (
            output
            + self.bias.data.reshape(
                1,
                self.out_channels,
                1,
                1
            )
        )

        expected_shape = (
            x.shape[0],
            self.out_channels,
            output_height,
            output_width
        )

        if output.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected convolution output shape. "
                f"Expected {expected_shape}, "
                f"received {tuple(output.shape)}."
            )

        return output

    def parameters(self):

        return [
            self.weight,
            self.bias
        ]