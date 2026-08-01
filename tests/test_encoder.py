import torch

from core.encoder import Encoder


encoder = Encoder()

image = torch.randn(
    2,
    1,
    64,
    64
)

latent = encoder(image)

print(latent.shape)