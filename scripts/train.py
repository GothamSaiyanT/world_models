import os

from torch.utils.data import Subset

from training.data_collector import DataCollector
from training.dataset import WorldModelSequenceDataset
from training.trainer import Trainer
from core.world_model import WorldModel


def main():

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists("data/frames.npy"):

        collector = DataCollector()

        collector.collect(
            num_steps=10000
        )

    full_dataset = WorldModelSequenceDataset(
        folder="data",
        seq_len=16
    )

    # -------------------------------------------------
    # CONTIGUOUS 80/20 SPLIT
    # -------------------------------------------------

    # Training sequences use frames from approximately
    # frame 0 to frame 7999.
    train_indices = list(
        range(0, 7984)
    )

    # Validation begins at frame 8000.
    # We leave a small sequence gap so a training
    # sequence does not cross into validation frames.
    validation_indices = list(
        range(8000, len(full_dataset))
    )

    train_dataset = Subset(
        full_dataset,
        train_indices
    )

    validation_dataset = Subset(
        full_dataset,
        validation_indices
    )

    print(
        "Training sequences:",
        len(train_dataset)
    )

    print(
        "Validation sequences:",
        len(validation_dataset)
    )

    model = WorldModel(
        latent_size=128,
        hidden_size=128,
        image_size=64
    )

    checkpoint = (
        "models/best_world_model.npz"
    )

    if os.path.exists(checkpoint):

        os.remove(checkpoint)

        print(
            "Old checkpoint deleted."
        )

    best_validation_loss = float(
        "inf"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        learning_rate=0.001,
        batch_size=32
    )

    # For tonight's first run
    epochs = 20

    print("Starting fresh training.")
    print("Epoch target:", epochs)

    for epoch in range(epochs):

        train_loss = trainer.train_epoch()

        validation_loss = (
            trainer.validate_epoch()
        )

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print(
            f"Training Loss:   "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.6f}"
        )

        # Save based on VALIDATION performance
        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            model.save(
                checkpoint,
                epoch=epoch + 1,
                best_loss=best_validation_loss
            )

            print(
                "Best validation model saved."
            )

    print(
        "\nTraining Complete!"
    )

    print(
        "Best Validation Loss:",
        f"{best_validation_loss:.6f}"
    )


if __name__ == "__main__":
    main()