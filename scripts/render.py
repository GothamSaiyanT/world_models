import argparse
import os
import torch

from training.dataset import WorldModelDataset
from utils.video import render_comparison_video
from core.rolloutgenerator import RolloutGenerator


CHECKPOINTS = {
    "baseline": "models/best_world_model.npz",
    "adaptive": "models/best_adaptive_model.pt",
    "fixed_interval": "models/best_fixed_interval_model.pt",
}


def load_model(pipeline, checkpoint, device):
    """
    Returns a model of the right class, loaded from the right
    checkpoint format, for the requested pipeline. This is the only
    part of the script that differs between pipelines -- the
    dataset, rollout logic, and video rendering are all shared.
    """

    if pipeline == "baseline":

        from core.world_model import WorldModel

        model = WorldModel(
            latent_size=128,
            hidden_size=128,
            image_size=64
        )

        result = model.load(checkpoint)

        # Baseline's custom load() may or may not return
        # (epoch, best_loss) depending on how you last saved it --
        # handle both cases so this script doesn't break either way.
        if isinstance(result, tuple):
            epoch, best_loss = result
            print(f"Checkpoint epoch: {epoch}, best loss: {best_loss}")

    else:

        from core_nn.world_model import WorldModel

        model = WorldModel(
            latent_size=128,
            hidden_size=128,
            image_size=64
        )

        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline",
        choices=["baseline", "adaptive", "fixed_interval"],
        default="baseline"
    )

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--fps", type=int, default=100)

    args = parser.parse_args()

    checkpoint = CHECKPOINTS[args.pipeline]

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Pipeline: {args.pipeline}")
    print("Rendering on:", device)

    dataset = WorldModelDataset(folder="data")

    if args.start < 0:
        raise ValueError("start cannot be negative.")

    if args.start + args.horizon >= len(dataset):
        raise ValueError(
            f"Requested frames up to index "
            f"{args.start + args.horizon}, but the dataset "
            f"contains only {len(dataset)} samples."
        )

    model = load_model(args.pipeline, checkpoint, device)

    initial_frame = (
        dataset[args.start][0]
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    actions = torch.stack([
        torch.as_tensor(dataset[args.start + i][1])
        for i in range(args.horizon)
    ]).to(device=device, dtype=torch.long).reshape(-1)

    real_sequence = torch.stack([
        dataset[args.start + i][0]
        for i in range(args.horizon + 1)
    ])

    print("Initial frame shape:", initial_frame.shape)
    print("Actions shape:", actions.shape)
    print("Real sequence shape:", real_sequence.shape)

    rollout = RolloutGenerator(model)

    predicted_sequence = rollout.generate(
        initial_frame=initial_frame,
        actions=actions
    )

    print("Predicted sequence shape:", predicted_sequence.shape)

    os.makedirs("outputs", exist_ok=True)

    output_path = f"outputs/rollout_comparison_{args.pipeline}.mp4"

    render_comparison_video(
        real_sequence,
        predicted_sequence,
        output_path,
        fps=args.fps
    )

    print(f"\nSaved comparison video to {output_path}")


if __name__ == "__main__":
    main()