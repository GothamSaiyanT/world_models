import torch

from core.decoder import Decoder


decoder = Decoder()

latent = torch.randn(
    2,
    128
)

image = decoder(latent)

print(image.shape)