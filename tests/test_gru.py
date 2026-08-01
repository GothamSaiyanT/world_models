import torch

from core.gru_cell import GRUCell


gru = GRUCell(
    input_size=160,
    hidden_size=128
)

x = torch.randn(
    5,
    160
)

hidden = torch.zeros(
    5,
    128
)

new_hidden = gru(
    x,
    hidden
)

print(new_hidden.shape)