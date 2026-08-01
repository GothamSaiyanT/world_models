import torch

from core.world_model import WorldModel
from training.dataset import WorldModelDataset
from utils.visualization import show_prediction


def main():

    dataset = WorldModelDataset(
        folder="data"
    )

    model = WorldModel()

    model.load(
        "models/best_world_model.npz"
    )

    model.eval()

    current_frame, action, target = dataset[0]

    current_frame = current_frame.unsqueeze(0)

    action = action.unsqueeze(0)

    hidden = model.init_hidden(
        batch_size=1
    )

    with torch.no_grad():

        prediction, _ = model(
            current_frame,
            action,
            hidden
        )

    show_prediction(
        current_frame,
        prediction,
        target
    )


if __name__ == "__main__":

    main()