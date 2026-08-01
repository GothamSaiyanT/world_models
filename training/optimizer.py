import torch


class SGD:

    def __init__(self, parameters, learning_rate=0.001):

        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def zero_grad(self):

        for parameter in self.parameters:

            parameter.zero_grad()

    def step(self):

        with torch.no_grad():

            for parameter in self.parameters:
                
                            print(
                                    "Leaf:",
                                    parameter.data.is_leaf,
                                    "Grad:",
                                    parameter.data.grad is not None,
                                    "Shape:",
                                    parameter.data.shape
                                   )

                if parameter.data.grad is None:
                    continue

                parameter.data -= (
                    self.learning_rate *
                    parameter.data.grad
                )