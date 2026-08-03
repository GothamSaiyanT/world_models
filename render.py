import os
import torch

from core.world_model import WorldModel
from core.rolloutgenerator import RolloutGenerator
from training.dataset import WorldModelDataset
from utils.video import render_comparison_video


def main(start=0, horizon=100, fps=10):

    dataset = WorldModelDataset(folder="data")

    model = WorldModel()
    model.load("models/best_world_model.npz")
    model.eval()

    initial_frame = dataset[start][0].unsqueeze(0)

    actions = torch.tensor([
        dataset[start + i][1] for i in range(horizon)
    ])

    real_sequence = [
        dataset[start + i][0] for i in range(horizon + 1)
    ]

    rollout = RolloutGenerator(model)

    predicted_sequence = rollout.generate(
        initial_frame,
        actions
    )

    os.makedirs("outputs", exist_ok=True)

    render_comparison_video(
        real_sequence,
        predicted_sequence,
        "outputs/rollout_comparison.mp4",
        fps=fps
    )


if __name__ == "__main__":
    main()