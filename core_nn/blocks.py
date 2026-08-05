import torch
from torch import nn


class Encoder(nn.Module):
    """Convert an observation frame into a compact latent representation."""

    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 64,
        input_width: int = 64,
        latent_size: int = 128,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            flattened_size = self.features(dummy).flatten(start_dim=1).shape[1]

        self.projection = nn.Linear(flattened_size, latent_size)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        features = self.features(frame)
        return self.projection(features.flatten(start_dim=1))


class Decoder(nn.Module):
    """Reconstruct an observation frame from a recurrent hidden state."""

    def __init__(
        self,
        latent_size: int = 128,
        image_size: int = 64,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.output_channels = output_channels

        output_size = output_channels * image_size * image_size
        self.network = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.network(latent)
        return decoded.view(
            latent.shape[0],
            self.output_channels,
            self.image_size,
            self.image_size,
        )
