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

        self.model.train()

        total_loss = 0.0
        number_of_batches = 0

        total_batches = len(
            self.dataloader
        )

        for batch_index, (frames, actions) in enumerate(
            self.dataloader
        ):

            if batch_index == 0:
                print(
                    f"Batch 1/{total_batches} loaded. "
                    "Starting calculations...",
                    flush=True
                )

            frames = frames.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True
            )

            actions = actions.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True
            )

            batch_size, sequence_length = actions.shape

            hidden = self.model.init_hidden(
                batch_size=batch_size
            ).to(self.device)

            self.optimizer.zero_grad()

            sequence_loss = torch.zeros(
                (),
                device=self.device,
                dtype=torch.float32
            )

            for timestep in range(sequence_length):

                current_frame = frames[
                    :,
                    timestep
                ]

                target_frame = frames[
                    :,
                    timestep + 1
                ]

                current_action = actions[
                    :,
                    timestep
                ]

                prediction, hidden = self.model(
                    current_frame,
                    current_action,
                    hidden
                )

                step_loss = self.criterion(
                    prediction=prediction,
                    target=target_frame,
                    current_frame=current_frame
                )

                sequence_loss = (
                    sequence_loss
                    + step_loss
                )

            sequence_loss = (
                sequence_loss
                / sequence_length
            )

            sequence_loss.backward()

            gradient_norm = clip_gradient_norm(
                self.parameters,
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += sequence_loss.item()
            number_of_batches += 1

            if (
                batch_index == 0
                or (batch_index + 1) % 10 == 0
                or batch_index + 1 == total_batches
            ):

                average_loss = (
                    total_loss
                    / number_of_batches
                )

                print(
                    f"Batch {batch_index + 1}/{total_batches} "
                    f"Average loss: {average_loss:.6f} "
                    f"Gradient norm: {gradient_norm:.4f}",
                    flush=True
                )

        if number_of_batches == 0:
            raise RuntimeError(
                "The DataLoader produced no batches."
            )

        return (
            total_loss
            / number_of_batches
        )