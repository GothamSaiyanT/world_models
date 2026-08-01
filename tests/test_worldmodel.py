import torch

from core.world_model import WorldModel


model = WorldModel()

frame = torch.randn(
    2,
    1,
    64,
    64
)

action = torch.randint(
    0,
    4,
    (2,)
)

hidden = model.init_hidden(2)

prediction, hidden = model(
    frame,
    action,
    hidden
)

print(prediction.shape)

print(hidden.shape)