import torch

from core.world_model import WorldModel
from training.dataset import WorldModelDataset
from utils.visualization import show_prediction


def main():

    print("Loading dataset...")

    dataset = WorldModelDataset(folder="data")

    print("Loading model...")

    model = WorldModel()

    model.load("models/best_world_model.npz")

    model.eval()

    print("Model loaded.")

    current_frame, action, target = dataset[0]

    current_frame = current_frame.unsqueeze(0)
    action = action.unsqueeze(0)

    hidden = model.init_hidden(1)

    print("Running prediction...")

    with torch.no_grad():
        prediction, _ = model(
            current_frame,
            action,
            hidden
        )

    print("Prediction complete.")

    show_prediction(
        current_frame,
        prediction,
        target
    )

    print("Visualization complete.")


if __name__ == "__main__":
    main()