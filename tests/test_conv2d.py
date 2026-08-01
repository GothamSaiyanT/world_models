import torch

from core.conv2d import Conv2D


conv = Conv2D(
    in_channels=1,
    out_channels=8,
    kernel_size=3
)

image = torch.randn(
    2,
    1,
    64,
    64
)

output = conv(image)

print(output.shape)