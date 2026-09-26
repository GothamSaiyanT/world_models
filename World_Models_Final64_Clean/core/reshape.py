from core.module import Module


class Reshape(Module):

    def __init__(self, *shape):

        super().__init__()

        self.shape = shape

    def forward(self, x):

        return x.reshape(*self.shape)