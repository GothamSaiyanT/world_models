import torch

from core.dynamics import Dynamics


model = Dynamics()

latent = torch.randn(
    4,
    128
)

action = torch.randint(
    0,
    4,
    (4,)
)

hidden = model.init_hidden(4)

output = model(
    latent,
    action,
    hidden
)

print(output.shape)