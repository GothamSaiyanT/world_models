import torch

from core.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit
    f(x) = max(0, x)
    """

    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):

        self.input = x

        return torch.maximum(
            x,
            torch.zeros_like(x)
        )

    def backward(self, grad_output):

        grad = grad_output.clone()

        grad[self.input <= 0] = 0

        return grad


class Sigmoid(Module):
    """
    Sigmoid Activation
    """

    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):

        self.output = 1.0 / (
            1.0 + torch.exp(-x)
        )

        return self.output

    def backward(self, grad_output):

        return (
            grad_output
            * self.output
            * (1 - self.output)
        )


class Tanh(Module):
    """
    Hyperbolic Tangent
    """

    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):

        self.output = torch.tanh(x)

        return self.output

    def backward(self, grad_output):

        return (
            grad_output
            * (1 - self.output ** 2)
        )