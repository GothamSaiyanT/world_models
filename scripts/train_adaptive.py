import os
import torch

from training.dataset import WorldModelSequenceDataset
from training.self_correcting_trainer import SelfCorrectingTrainer

from core_nn.world_model import WorldModel
from core_nn.drift import DriftDetector
from core_nn.corrector import AdaptiveCorrector


def main():

    # -----------------------
    # Load dataset
    # -----------------------
    # Reuses the exact same frames.npy / actions.npy collected for
    # the baseline -- all three pipelines must train on identical
    # data for the comparison to be valid.

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

    drift_detector = DriftDetector(
        metric="mse",
        threshold=0.01   # tune this: see note below
    )

    corrector = AdaptiveCorrector(
        threshold=drift_detector.threshold
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

    for epoch in range(epochs):

        loss = trainer.train_epoch()

        num_corrections = len(corrector.correction_log)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss:.6f} "
            f"Corrections this epoch: {num_corrections}"
        )

        # Warn if the threshold is clearly miscalibrated -- this
        # mirrors the "too sensitive / too loose" extensions from
        # the Deliverable 3 use case description.
        if num_corrections == 0:
            print(
                "  Warning: no corrections fired this epoch -- "
                "the threshold may be too loose."
            )

        if loss < best_loss:

            best_loss = loss

            torch.save(
                model.state_dict(),
                "models/best_adaptive_model.pt"
            )

            print("  Best adaptive model saved.")

    print("\nTraining Complete!")
    print(f"Best Loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
