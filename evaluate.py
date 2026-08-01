import torch

from core.world_model import WorldModel
from training.dataset import WorldModelDataset
from core.loss import MSELoss


def main():

    dataset = WorldModelDataset(
        folder="data"
    )

    model = WorldModel()

    model.load(
        "models/best_world_model.npz"
    )

    model.eval()

    criterion = MSELoss()

    total_loss = 0

    with torch.no_grad():

        for i in range(len(dataset)):

            current_frame, action, target = dataset[i]

            current_frame = current_frame.unsqueeze(0)
            action = action.unsqueeze(0)

            hidden = model.init_hidden(
                batch_size=1
            )

            prediction, _ = model(
                current_frame,
                action,
                hidden
            )

            loss = criterion(
                prediction,
                target.unsqueeze(0)
            )

            total_loss += loss.item()

    average_loss = total_loss / len(dataset)

    print("Evaluation Complete")
    print(f"Samples: {len(dataset)}")
    print(f"Average Loss: {average_loss:.6f}")


if __name__ == "__main__":
    main()