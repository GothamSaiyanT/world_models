"""Render a 128x128 real-vs-predicted rollout for any trained pipeline."""

import argparse
import os
import torch

from config import ModelConfig
from highres_128 import DATA_FOLDER, IMAGE_SIZE, MODEL_FOLDER, OUTPUT_FOLDER
from training.dataset import WorldModelDataset
from utils.rollout_generator import RolloutGenerator
from utils.video import render_comparison_video


def load_model(pipeline: str, checkpoint: str, device: torch.device):
    if pipeline == "baseline":
        from core.world_model import WorldModel
        model = WorldModel(latent_size=128, hidden_size=128, image_size=IMAGE_SIZE)
        epoch, best_loss = model.load(checkpoint)
        print(f"Checkpoint epoch: {epoch}, best loss: {best_loss}")
    else:
        from core_nn.world_model import WorldModel
        config = ModelConfig(image_size=IMAGE_SIZE)
        model = WorldModel(config)
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "adaptive", "fixed_interval"],
        default="baseline",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--model-folder", default=MODEL_FOLDER)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER)
    args = parser.parse_args()

    checkpoints = {
        "baseline": os.path.join(args.model_folder, "best_world_model.npz"),
        "adaptive": os.path.join(args.model_folder, "best_adaptive_model.pt"),
        "fixed_interval": os.path.join(args.model_folder, "best_fixed_interval_model.pt"),
    }
    checkpoint = checkpoints[args.pipeline]
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WorldModelDataset(folder=args.data_folder)

    if args.start < 0 or args.horizon < 1:
        raise ValueError("start must be >= 0 and horizon must be >= 1.")
    if args.start + args.horizon >= len(dataset):
        raise ValueError(
            f"start+horizon={args.start + args.horizon}, but dataset has "
            f"only {len(dataset)} usable samples."
        )

    model = load_model(args.pipeline, checkpoint, device)
    initial_frame = dataset[args.start][0].unsqueeze(0).to(device)
    actions = torch.stack([dataset[args.start + i][1] for i in range(args.horizon)])
    actions = actions.to(device=device, dtype=torch.long).reshape(-1)
    real_sequence = torch.stack([
        dataset[args.start + i][0] for i in range(args.horizon + 1)
    ])

    prediction = RolloutGenerator(model).generate(initial_frame, actions)

    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(
        args.output_folder, f"rollout_comparison_{args.pipeline}.mp4"
    )
    render_comparison_video(real_sequence, prediction, output_path, fps=args.fps)
    print("Initial frame:", tuple(initial_frame.shape))
    print("Prediction:", tuple(prediction.shape))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
