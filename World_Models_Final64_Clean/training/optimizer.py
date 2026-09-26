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

                if parameter.data.grad is None:
                    continue

                parameter.data -= (
                    self.learning_rate *
                    parameter.data.grad
                )


class Adam:

    def __init__(
        self,
        parameters,
        learning_rate=0.0003,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.parameters = list(parameters)
        self.learning_rate = learning_rate

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Timestep, used for bias correction
        self.t = 0

        # First and second moment estimates, one per parameter
        self.m = [
            torch.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

        self.v = [
            torch.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

    def zero_grad(self):

        for parameter in self.parameters:

            parameter.zero_grad()

    def step(self):

        self.t += 1

        with torch.no_grad():

            for index, parameter in enumerate(self.parameters):

                if parameter.data.grad is None:
                    continue

                gradient = parameter.data.grad

                # Update biased first moment estimate (momentum)
                self.m[index] = (
                    self.beta1 * self.m[index]
                    + (1 - self.beta1) * gradient
                )

                # Update biased second moment estimate (variance)
                self.v[index] = (
                    self.beta2 * self.v[index]
                    + (1 - self.beta2) * (gradient ** 2)
                )

                # Bias-corrected estimates
                m_hat = (
                    self.m[index]
                    / (1 - self.beta1 ** self.t)
                )

                v_hat = (
                    self.v[index]
                    / (1 - self.beta2 ** self.t)
                )

                parameter.data -= (
                    self.learning_rate
                    * m_hat
                    / (torch.sqrt(v_hat) + self.epsilon)
                )