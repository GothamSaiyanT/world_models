import torch

from core.module import Module
from core.gru_cell import GRUCell
from core.embedding import Embedding

class Dynamics(Module):

    def __init__(self,
                 latent_size=128,
                 hidden_size=128):

        super().__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_embedding = Embedding(
            num_embeddings = 4,
            embedding_dim = 32
        )

        self.gru = GRUCell(
            input_size=latent_size + 32,
            hidden_size=hidden_size
        )

    def init_hidden(self, batch_size):

        return torch.zeros(
            batch_size,
            self.hidden_size
        )

    def forward(self, latent,action , hidden):

        action_vector = self.action_embedding(
            action
        )
        gru_input = torch.cat(
            (
            latent,
            action_vector
            ),
            dim=1
        )
        hidden = self.gru(
            gru_input,
            hidden
        )

        return hidden

    def parameters(self):

        params = []

        params.extend(
            self.action_embedding.parameters()
        )

        params.extend(
            self.gru.parameters()
        )

        return params

    