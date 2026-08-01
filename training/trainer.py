import torch
from torch.utils.data import DataLoader

from core.loss import MSELoss
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

        self.criterion = MSELoss()

        self.optimizer = SGD(
            model.parameters(),
            learning_rate
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        for batch_index, (
            current_frame,
            action,
            next_frame
        ) in enumerate(self.dataloader):

            current_frame = current_frame.to(
                self.device,
                non_blocking=True
            )

            next_frame = next_frame.to(
                self.device,
                non_blocking=True
            )

            action = action.to(
                self.device,
                non_blocking=True
            )

            hidden = self.model.init_hidden(
                current_frame.size(0)
            ).to(self.device)

            self.optimizer.zero_grad()

            prediction, hidden = self.model(
                current_frame,
                action,
                hidden
            )

            loss = self.criterion(
                prediction,
                next_frame
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            if batch_index % 20 == 0:
                print(
                    f"Batch {batch_index}/{len(self.dataloader)} "
                    f"Loss: {loss.item():.6f}"
                )

        return total_loss / len(self.dataloader)