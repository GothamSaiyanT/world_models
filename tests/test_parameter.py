import torch

from core.parameter import Parameter


parameter = Parameter(
    torch.randn(3,4)
)

print(parameter)

print(parameter.shape)

print(parameter.ndim)

print(parameter.dtype)