import torch

from core.embedding import Embedding


embedding = Embedding(
    num_embeddings=4,
    embedding_dim=32
)

actions = torch.tensor(
    [0,1,2,3]
)

output = embedding(actions)

print(output.shape)