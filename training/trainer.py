import torch
from torch.utils.data import DataLoader

from core.loss import AdaptiveMotionWeightedMSELoss
from training.optimizer import SGD


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        learning_rate=0.001,
        batch_size=32
    ):

        self.model = model

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Training on: {self.device}")

        # Move every parameter to the device
        self.model.to(self.device)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        self.criterion = AdaptiveMotionWeightedMSELoss(
        motion_weight=10.0,
        motion_threshold=0.02,
        base_loss_weight=1.0,
        max_motion_fraction=0.35
            )
        self.optimizer = SGD(
            model.parameters(),
            learning_rate
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0
        number_of_batches = 0

        total_batches = len(self.dataloader)

        for batch_index, (frames, actions) in enumerate(
            self.dataloader
        ):

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

                current_frame = frames[:, timestep]
                target_frame = frames[:, timestep + 1]
                current_action = actions[:, timestep]

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
                    sequence_loss + step_loss
                )

            sequence_loss = (
                sequence_loss / sequence_length
            )

            sequence_loss.backward()

            self.optimizer.step()

            total_loss += sequence_loss.item()
            number_of_batches += 1

            # Show progress every 10 batches
            if (
                batch_index == 0
                or (batch_index + 1) % 10 == 0
                or batch_index + 1 == total_batches
            ):

                average_loss = (
                    total_loss / number_of_batches
                )

                print(
                    f"Batch {batch_index + 1}/{total_batches} "
                    f"Average loss: {average_loss:.6f}",
                    flush=True
                )

        if number_of_batches == 0:
            raise RuntimeError(
                "The DataLoader produced no batches."
            )

        return total_loss / number_of_batches