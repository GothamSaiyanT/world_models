from __future__ import annotations

import torch
from torch import nn

from v3.config import V3Config


class ConvEncoder128(nn.Module):
    """128x128 grayscale frame -> compact latent vector."""

    def __init__(self, config: V3Config):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, 4, 2, 1),   # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),                      # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),                     # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),                    # 16 -> 8
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, config.latent_size),
            nn.LayerNorm(config.latent_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(frame))


class ConvDecoder128(nn.Module):
    """Recurrent hidden state -> 128x128 grayscale frame."""

    def __init__(self, config: V3Config):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(config.hidden_size, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),            # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),             # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),              # 32 -> 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, config.input_channels, 4, 2, 1), # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.input(hidden)
        x = x.view(hidden.shape[0], 256, 8, 8)
        return self.decoder(x)


class WorldModelV3(nn.Module):
    """
    Shared model used by Baseline, Fixed-Interval, and Adaptive V3.

    All three experiments therefore have the same architecture. The only
    experimental difference is the observation-correction strategy.
    """

    def __init__(self, config: V3Config | None = None):
        super().__init__()
        self.config = config or V3Config()
        if self.config.image_size != 128:
            raise ValueError("WorldModelV3 currently expects image_size=128.")
        if self.config.latent_size != self.config.hidden_size:
            raise ValueError(
                "V3 uses encoded real observations to reset recurrent state; "
                "latent_size must equal hidden_size."
            )

        self.encoder = ConvEncoder128(self.config)
        self.action_embedding = nn.Embedding(
            self.config.num_actions,
            self.config.action_embedding_size,
        )
        self.gru = nn.GRUCell(
            self.config.latent_size + self.config.action_embedding_size,
            self.config.hidden_size,
        )
        # Explicit residual action path. Motion-focused supervision still has
        # to learn the magnitude, but action information cannot only enter at
        # the encoder boundary.
        self.action_to_hidden = nn.Sequential(
            nn.Linear(self.config.action_embedding_size, self.config.hidden_size),
            nn.Tanh(),
        )
        self.decoder = ConvDecoder128(self.config)

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
        next_hidden = next_hidden + 0.10 * self.action_to_hidden(action_vector)
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
