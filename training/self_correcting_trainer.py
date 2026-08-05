import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SelfCorrectingTrainer:
    """
    Same overall shape as training/trainer.py (unrolls over
    sequences, carries hidden state across steps), but with two
    differences that make it "self-correcting":

    1. The input fed forward at each step is the model's OWN
       prediction from the previous step, not the real frame. This
       is what actually exposes the model to compounding drift
       during training, rather than only at evaluation time.

    2. After each step, the given Corrector is given the chance to
       intervene on the hidden state using the real frame as an
       anchor, based on the error measured by DriftDetector.

    The same class is used for both the adaptive and fixed-interval
    pipelines -- they differ only in which Corrector instance is
    passed in.
    """

    def __init__(
        self,
        model,
        dataset,
        corrector,
        drift_detector,
        learning_rate=0.001,
        batch_size=32
    ):

        self.model = model
        self.corrector = corrector
        self.drift_detector = drift_detector

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Training on: {self.device}")

        self.model.to(self.device)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate
        )

    def train_epoch(self):

        self.model.train()

        self.corrector.reset_log()

        total_loss = 0.0

        for frames, actions in self.dataloader:

            frames = frames.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)

            batch_size, seq_len = actions.shape

            hidden = self.model.init_hidden(batch_size, self.device)

            self.optimizer.zero_grad()

            # First input is the real starting frame; after that,
            # the model feeds its own predictions forward.
            frame = frames[:, 0]

            sequence_loss = 0.0

            for t in range(seq_len):

                latent = self.model.encode(frame)

                prediction, hidden = self.model.step(
                    latent,
                    actions[:, t],
                    hidden
                )

                real_next = frames[:, t + 1]

                sequence_loss = sequence_loss + F.mse_loss(
                    prediction,
                    real_next
                )

                with torch.no_grad():
                    error = self.drift_detector.compute_error(
                        prediction,
                        real_next
                    )

                hidden = self.corrector.maybe_correct(
                    hidden,
                    real_next,
                    self.model.encode,
                    error,
                    t
                )

                # Feed the prediction forward autoregressively.
                # Detached so gradients only flow through the
                # current step, not back through the whole chain
                # of predictions (keeps training stable/cheap).
                frame = prediction.detach()

            sequence_loss = sequence_loss / seq_len

            sequence_loss.backward()

            self.optimizer.step()

            total_loss += sequence_loss.item()

        return total_loss / len(self.dataloader)
