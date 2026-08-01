import torch

from core.flatten import Flatten

flatten = Flatten()

x = torch.randn(2,3,4,4)

print("input shape:",x.shape)

y = flatten.forward(x)

print("flatten Shape:",y.shape)

z = flatten.backward(y)

print("recovered shape:",z.shape)