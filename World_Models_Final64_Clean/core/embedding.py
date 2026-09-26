import torch

from core.module import Module
from core.parameter import Parameter


class Embedding(Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim):

        super().__init__()

        self.weight = Parameter(
            torch.randn(
                num_embeddings,
                embedding_dim
            ) * 0.01
        )

    def forward(self, indices):

        return self.weight.data[indices]

    def parameters(self):

        return [self.weight]