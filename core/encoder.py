import torch

from core.conv2d import Conv2D
from core.activation import ReLU
from core.flatten import Flatten
from core.linear import Linear
from core.module import Module


class Encoder(Module):

    def __init__(
        self,
        input_channels=1,
        input_height=64,
        input_width=64,
        latent_size=128
    ):
        super().__init__()

        # -----------------------------
        # Convolution Layers
        # -----------------------------

        self.conv1 = Conv2D(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=0
        )

        self.relu1 = ReLU()

        self.conv2 = Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0
        )

        self.relu2 = ReLU()

        self.conv3 = Conv2D(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=0
        )

        self.relu3 = ReLU()

        # -----------------------------
        # Flatten Layer
        # -----------------------------

        self.flatten = Flatten()

        # Automatically calculate the size after convolutions

        flatten_size = self._get_flatten_size(
            input_channels,
            input_height,
            input_width
        )

        # -----------------------------
        # Latent Projection
        # -----------------------------

        self.linear = Linear(
            in_features=flatten_size,
            out_features=latent_size
        )

    def _get_flatten_size(
        self,
        input_channels,
        input_height,
        input_width
    ):

        dummy = torch.zeros(
            1,
            input_channels,
            input_height,
            input_width
        )

        dummy = self.conv1(dummy)
        dummy = self.relu1(dummy)

        dummy = self.conv2(dummy)
        dummy = self.relu2(dummy)

        dummy = self.conv3(dummy)
        dummy = self.relu3(dummy)

        dummy = self.flatten(dummy)

        return dummy.shape[1]

    def forward(self, x):
        """
        Encode an image into a latent representation.

        Input:
            (batch_size, channels, height, width)

        Output:
            (batch_size, latent_size)
        """

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.flatten(x)

        latent = self.linear(x)

        return latent

    def parameters(self):

        params = []

        params.extend(
            self.conv1.parameters()
        )

        params.extend(
            self.conv2.parameters()
        )

        params.extend(
            self.conv3.parameters()
        )

        params.extend(
            self.linear.parameters()
        )

        return params

    def children(self):

        return [
            self.conv1,
            self.relu1,
            self.conv2,
            self.relu2,
            self.conv3,
            self.relu3,
            self.flatten,
            self.linear
        ]