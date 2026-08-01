import os

from training.data_collector import DataCollector
from training.dataset import WorldModelDataset
from training.trainer import Trainer

from core.world_model import WorldModel


def main():

    # -----------------------
    # Collect dataset once
    # -----------------------

    if not os.path.exists("data/frames.npy"):

        collector = DataCollector()

        collector.collect(
            num_steps=10000
        )

    # -----------------------
    # Load dataset
    # -----------------------

    dataset = WorldModelDataset(
        folder="data"
    )

    # -----------------------
    # Create model
    # -----------------------

    model = WorldModel(
        latent_size=128,
        hidden_size=128,
        image_size=64
    )

    # -----------------------
    # Trainer
    # -----------------------

    trainer = Trainer(
        model=model,
        dataset=dataset,
        learning_rate=0.001,
        batch_size=32
    )

    epochs = 1
    print("DEBUG EPOCH VALUE:", epochs)
    best_loss = float("inf")

    os.makedirs(
        "models",
        exist_ok=True
    )

    # -----------------------
    # Training Loop
    # -----------------------

    for epoch in range(epochs):

        loss = trainer.train_epoch()

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" Loss: {loss:.6f}"
        )

        if loss < best_loss:

            best_loss = loss

            model.save(
                "models/best_world_model.npz"
            )

            print(
                "Best model saved."
            )

    print("\nTraining Complete!")

    print(
        f"Best Loss: {best_loss:.6f}"
    )


if __name__ == "__main__":

    main()