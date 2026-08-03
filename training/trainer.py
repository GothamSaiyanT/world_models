import torch
from torch.utils.data import DataLoader

from core.loss import StableMotionWeightedMSELoss
from training.optimizer import Adam


def clip_gradient_norm(
    parameters,
    max_norm=1.0
):
    """
    Clips the combined gradient norm of all parameters.

    This implementation does not use torch.nn utilities.
    """

    gradients = []

    for parameter in parameters:

        if parameter.data.grad is not None:
            gradients.append(
                parameter.data.grad
            )

    if len(gradients) == 0:
        return 0.0

    total_squared_norm = torch.zeros(
        (),
        device=gradients[0].device,
        dtype=gradients[0].dtype
    )

    for gradient in gradients:

        total_squared_norm = (
            total_squared_norm
            + (gradient.detach() ** 2).sum()
        )

    total_norm = torch.sqrt(
        total_squared_norm
    )

    scale = torch.clamp(
        max_norm / (total_norm + 1e-8),
        max=1.0
    )

    for gradient in gradients:
        gradient.mul_(scale)

    return total_norm.item()


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        learning_rate=0.0003,
        batch_size=32
    ):

        self.model = model

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Training on: {self.device}")

        # Move all custom model parameters to the selected device
        self.model.to(self.device)

        self.parameters = list(
            self.model.parameters()
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(
                self.device.type == "cuda"
            )
        )

        self.criterion = StableMotionWeightedMSELoss(
            motion_weight=2.0,
            motion_threshold=0.05
        )

        self.optimizer = Adam(
            self.parameters,
            learning_rate
        )

    def train_epoch(self):

        # ... unchanged, exactly as before ...