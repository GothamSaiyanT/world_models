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
        """
        Each batch is now (frames, actions), where:
            frames:  (batch, seq_len + 1, 1, H, W)
            actions: (batch, seq_len)

        The hidden state is initialised once per sequence and
        carried forward across every step of that sequence, instead
        of being reset to zero every batch. This is what actually
        lets the GRU learn temporal structure (e.g. ball direction),
        which was missing before and is a likely cause of the noisy
        one-step predictions.
        """

        self.model.train()

        total_loss = 0.0

        for frames, actions in self.dataloader:

            frames = frames.to(
                self.device,
                non_blocking=True
            )

            actions = actions.to(
                self.device,
                non_blocking=True
            )

            batch_size, seq_len = actions.shape

            hidden = self.model.init_hidden(
                batch_size
            ).to(self.device)

            self.optimizer.zero_grad()

            sequence_loss = 0.0

            for t in range(seq_len):

                current_frame = frames[:, t]
                next_frame = frames[:, t + 1]
                action = actions[:, t]

                prediction, hidden = self.model(
                    current_frame,
                    action,
                    hidden
                )

                sequence_loss = sequence_loss + self.criterion(
                    prediction,
                    next_frame
                )

            sequence_loss = sequence_loss / seq_len

            sequence_loss.backward()

            self.optimizer.step()

            total_loss += sequence_loss.item()

        return total_loss / len(self.dataloader)