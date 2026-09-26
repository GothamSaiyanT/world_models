from __future__ import annotations

import torch
from torch import nn

from v4.config import V4Config


class ConvEncoder64(nn.Module):
    """64x64 grayscale frame -> latent vector."""

    def __init__(self, config: V4Config):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, 4, 2, 1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),                      # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),                     # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),                    # 8 -> 4
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, config.latent_size),
            nn.LayerNorm(config.latent_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(frame))


class UpsampleConvBlock(nn.Module):
    """Upsample then convolve; avoids transposed-convolution checkerboard artifacts."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvDecoder64(nn.Module):
    """Recurrent hidden state -> 64x64 grayscale frame."""

    def __init__(self, config: V4Config):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(config.hidden_size, 128 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            UpsampleConvBlock(128, 128),                     # 4 -> 8
            UpsampleConvBlock(128, 64),                      # 8 -> 16
            UpsampleConvBlock(64, 32),                       # 16 -> 32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, config.input_channels, 3, 1, 1),  # 32 -> 64
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.input(hidden)
        x = x.view(hidden.shape[0], 128, 4, 4)
        return self.decoder(x)


class WorldModelV4(nn.Module):
    """
    Balanced 64x64 world model.

    The architecture is shared by Baseline, Fixed-Interval and Adaptive.
    V4 keeps V3's action-conditioned recurrent dynamics but uses a smaller
    decoder and a more conservative motion objective to reduce ghost trails.
    """

    def __init__(self, config: V4Config | None = None):
        super().__init__()
        self.config = config or V4Config()
        if self.config.image_size != 64:
            raise ValueError("WorldModelV4 expects image_size=64.")
        if self.config.latent_size != self.config.hidden_size:
            raise ValueError(
                "latent_size must equal hidden_size because real observations "
                "can reset recurrent state during correction."
            )

        self.encoder = ConvEncoder64(self.config)
        self.action_embedding = nn.Embedding(
            self.config.num_actions,
            self.config.action_embedding_size,
        )
        self.gru = nn.GRUCell(
            self.config.latent_size + self.config.action_embedding_size,
            self.config.hidden_size,
        )
        self.action_to_hidden = nn.Sequential(
            nn.Linear(self.config.action_embedding_size, self.config.hidden_size),
            nn.Tanh(),
        )
        self.decoder = ConvDecoder64(self.config)

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(frame)

    def step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_vector = self.action_embedding(action.long())
        recurrent_input = torch.cat([latent, action_vector], dim=1)
        next_hidden = self.gru(recurrent_input, hidden)
        # Smaller residual action injection than V3 to reduce overshoot.
        next_hidden = next_hidden + 0.05 * self.action_to_hidden(action_vector)
        prediction = self.decoder(next_hidden)
        return prediction, next_hidden

    def forward(
        self,
        frame: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(self.encode(frame), action, hidden)

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.hidden_size, device=device)
