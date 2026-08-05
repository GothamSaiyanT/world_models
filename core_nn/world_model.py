import torch
from torch import nn

from world_model_oop.config import ModelConfig
from world_model_oop.core.blocks import Decoder, Encoder


class WorldModel(nn.Module):
    """Encoder -> action-conditioned GRU -> decoder world model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = Encoder(
            input_channels=config.input_channels,
            input_height=config.image_size,
            input_width=config.image_size,
            latent_size=config.latent_size,
        )
        self.action_embedding = nn.Embedding(
            config.num_actions,
            config.action_embedding_size,
        )
        self.gru = nn.GRUCell(
            input_size=config.latent_size + config.action_embedding_size,
            hidden_size=config.hidden_size,
        )
        self.decoder = Decoder(
            latent_size=config.hidden_size,
            image_size=config.image_size,
            output_channels=config.input_channels,
        )

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(frame)

    def step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_vector = self.action_embedding(action.long())
        gru_input = torch.cat((latent, action_vector), dim=1)
        next_hidden = self.gru(gru_input, hidden)
        prediction = self.decoder(next_hidden)
        return prediction, next_hidden

    def forward(
        self,
        frame: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(self.encode(frame), action, hidden)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)
