import torch

from core.linear import Linear


layer = Linear(
    in_features=5,
    out_features=3
)

x = torch.randn(
    2,
    5
)

y = layer(x)

print(y.shape)