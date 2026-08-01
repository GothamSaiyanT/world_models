import torch


class MSELoss:
    """
    Mean Squared Error Loss
    """

    def forward(self, prediction, target):

        loss = torch.mean(
            (prediction - target) ** 2
        )

        return loss

    def __call__(self, prediction, target):

        return self.forward(
            prediction,
            target
        )