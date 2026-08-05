import os
import torch

from training.dataset import WorldModelSequenceDataset
from training.self_correcting_trainer import SelfCorrectingTrainer

from core_nn.world_model import WorldModel
from core_nn.drift_detector import DriftDetector
from core_nn.correction import FixedIntervalCorrector


def main():

    # -----------------------
    # Load dataset (same data as baseline and adaptive pipelines)
    # -----------------------

    dataset = WorldModelSequenceDataset(
        folder="data",
        seq_len=16
    )

    # -----------------------
    # Create model + correction components
    # -----------------------

    model = WorldModel(
        latent_size=128,
        hidden_size=128,
        image_size=64
    )

    # DriftDetector is still used here, purely for logging/
    # comparison purposes (so you can report the error at the
    # moment each fixed-interval correction fires) -- it does not
    # influence *when* FixedIntervalCorrector corrects.
    drift_detector = DriftDetector(
        metric="mse",
        threshold=0.01
    )

    interval = 10   # correct every 10 steps; must be <= seq_len

    corrector = FixedIntervalCorrector(
        interval=interval
    )

    # -----------------------
    # Trainer
    # -----------------------

    trainer = SelfCorrectingTrainer(
        model=model,
        dataset=dataset,
        corrector=corrector,
        drift_detector=drift_detector,
        learning_rate=0.001,
        batch_size=32
    )

    epochs = 30
    best_loss = float("inf")

    os.makedirs("models", exist_ok=True)

    # Mirrors the "N bigger than the number of steps imagined" UC6
    # extension from Deliverable 3.
    if interval > dataset.seq_len:
        print(
            f"Warning: interval ({interval}) is larger than "
            f"seq_len ({dataset.seq_len}) -- no correction will "
            f"ever fire during training."
        )

    for epoch in range(epochs):

        loss = trainer.train_epoch()

        num_corrections = len(corrector.correction_log)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss:.6f} "
            f"Corrections this epoch: {num_corrections}"
        )

        if loss < best_loss:

            best_loss = loss

            torch.save(
                model.state_dict(),
                "models/best_fixed_interval_model.pt"
            )

            print("  Best fixed-interval model saved.")

    print("\nTraining Complete!")
    print(f"Best Loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
