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

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.criterion = MSELoss()

        self.optimizer = SGD(
            model.parameters(),
            learning_rate
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for batch_index, (current_frame, action, next_frame) in enumerate(self.dataloader):

            print(f"Batch {batch_index} loaded")

            hidden = self.model.init_hidden(
                current_frame.size(0)
            )

            print("Hidden initialized")

            self.optimizer.zero_grad()

            print("Forward pass...")

            prediction, hidden = self.model(
                current_frame,
                action,
                hidden
            )

            print("Forward complete")

            loss = self.criterion(
                prediction,
                next_frame
            )

            print("Loss computed")

            loss.backward()

            print("Backward complete")

            self.optimizer.step()

            print("Optimizer step complete")

            total_loss += loss.item()

        print("Epoch complete")

        return total_loss / len(self.dataloader)