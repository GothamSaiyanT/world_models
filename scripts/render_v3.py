from __future__ import annotations

import argparse
import os

import torch

from training.dataset import WorldModelDataset
from utils.rollout_v3 import RolloutGeneratorV3
from utils.video import render_comparison_video
from v3.config import V3Config
from v3.world_model import WorldModelV3


def load_checkpoint(pipeline, model_folder, device):
    model_path = os.path.join(model_folder, f"best_{pipeline}_model.pt")
    checkpoint_path = os.path.join(model_folder, f"best_{pipeline}_checkpoint.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = WorldModelV3(V3Config())
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    metadata = {}
    if os.path.exists(checkpoint_path):
        metadata = torch.load(checkpoint_path, map_location="cpu")
    return model, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "fixed_interval", "adaptive"],
        required=True,
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--data-folder", default="data_v3")
    parser.add_argument("--model-folder", default="models_v3")
    parser.add_argument("--output-folder", default="outputs_v3")
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--adaptive-threshold", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WorldModelDataset(args.data_folder)
    if args.start + args.horizon >= len(dataset):
        raise ValueError("Requested rollout exceeds dataset length.")

    model, metadata = load_checkpoint(args.pipeline, args.model_folder, device)
    initial = dataset[args.start][0].unsqueeze(0).to(device)
    actions = torch.stack([
        dataset[args.start + i][1] for i in range(args.horizon)
    ]).to(device).long()
    real = torch.stack([
        dataset[args.start + i][0] for i in range(args.horizon + 1)
    ])

    interval = args.interval or int(metadata.get("fixed_interval", 8))
    adaptive_threshold = args.adaptive_threshold
    if adaptive_threshold is None:
        adaptive_threshold = metadata.get("adaptive_threshold")
        if adaptive_threshold is None:
            adaptive_threshold = metadata.get("validation_p90_drift")

    generator = RolloutGeneratorV3(
        model,
        strategy=args.pipeline,
        fixed_interval=interval,
        adaptive_threshold=adaptive_threshold,
    )
    prediction = generator.generate(
        initial,
        actions,
        real_sequence=real if args.pipeline != "baseline" else None,
    )

    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(
        args.output_folder,
        f"{args.pipeline}_start{args.start}_h{args.horizon}.mp4",
    )
    render_comparison_video(real, prediction.cpu(), output_path, fps=args.fps)
    print("Saved:", output_path)
    print("Prediction shape:", tuple(prediction.shape))
    print("Correction steps:", generator.correction_steps[:30])
    print("Number of corrections:", len(generator.correction_steps))
    if args.pipeline == "adaptive":
        print("Adaptive threshold:", adaptive_threshold)

if __name__ == "__main__":
    main()
