import os

from training.data_collector import DataCollector
from training.dataset import WorldModelSequenceDataset
from training.trainer import Trainer
from core.world_model import WorldModel


def main():

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Keep the existing dataset.
    # Data is only collected when frames.npy does not exist.
    if not os.path.exists("data/frames.npy"):

        collector = DataCollector()

        collector.collect(
            num_steps=10000
        )

    dataset = WorldModelSequenceDataset(
        folder="data",
        seq_len=16
    )

    # Create a completely new model with random parameters
    model = WorldModel(
        latent_size=128,
        hidden_size=128,
        image_size=64
    )

    checkpoint = "models/best_world_model.npz"

    # Delete the previous model checkpoint
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
        print("Old checkpoint deleted.")

    # Always start from the beginning
    start_epoch = 0
    best_loss = float("inf")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        learning_rate=0.001,
        batch_size=32
    )

    epochs = 250

    print("Starting fresh training.")
    print("Epoch target:", epochs)

    for epoch in range(start_epoch, epochs):

        loss = trainer.train_epoch()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss:.6f}"
        )

        if loss < best_loss:

            best_loss = loss

            model.save(
                checkpoint,
                epoch=epoch + 1,
                best_loss=best_loss
            )

            print("Best model saved.")

    print("\nTraining Complete!")
    print(f"Best Loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()