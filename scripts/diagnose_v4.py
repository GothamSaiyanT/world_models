from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from training.dataset import WorldModelDataset
from utils.rollout_v4 import RolloutGeneratorV4
from v3.device import resolve_accelerator
from v4.config import V4Config
from v4.world_model import WorldModelV4


def departed_pixel_ghost_score(real, prediction, fg=0.04, bg=0.03):
    """
    Mean predicted brightness where a real foreground pixel disappeared.
    Lower is better. This directly measures old-position ghost persistence.
    """
    scores = []
    fractions = []
    for i in range(1, min(len(real), len(prediction))):
        previous = real[i - 1]
        target = real[i]
        pred = prediction[i]
        mask = (previous > fg) & (target <= bg)
        count = int(mask.sum())
        if count > 0:
            scores.append(float(pred[mask].mean()))
            fractions.append(float((pred[mask] > 0.10).float().mean()))
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.mean(fractions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="data_v4_64")
    parser.add_argument("--model-folder", default="models_v4_64")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "tpu", "cpu"],
    )
    args = parser.parse_args()

    frames = np.load(os.path.join(args.data_folder, "frames.npy"))
    actions_np = np.load(os.path.join(args.data_folder, "actions.npy"))
    motion = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2))

    print("===== DATASET =====")
    print("Frames:", frames.shape)
    print("Mean motion:", float(motion.mean()))
    print("Median motion:", float(np.median(motion)))
    print("P90 motion:", float(np.percentile(motion, 90)))
    values, counts = np.unique(actions_np, return_counts=True)
    for value, count in zip(values, counts):
        print(f"Action {value}: {count} ({100*count/len(actions_np):.2f}%)")

    runtime = resolve_accelerator(args.device)
    device = runtime.device
    print("Accelerator:", runtime.label)
    print("Evaluation start:", args.start)
    print("Evaluation horizon:", args.horizon)

    dataset = WorldModelDataset(args.data_folder)
    if args.start < 0 or args.start + args.horizon >= len(dataset):
        raise ValueError("Requested start/horizon exceeds dataset length.")

    initial = dataset[args.start][0].unsqueeze(0).to(device)
    actions = torch.stack([
        dataset[args.start + i][1]
        for i in range(args.horizon)
    ]).to(device)
    real = torch.stack([
        dataset[args.start + i][0]
        for i in range(args.horizon + 1)
    ])

    for pipeline in ["baseline", "fixed_interval", "adaptive"]:
        model_path = os.path.join(
            args.model_folder,
            f"best_{pipeline}_model.pt",
        )
        checkpoint_path = os.path.join(
            args.model_folder,
            f"best_{pipeline}_checkpoint.pt",
        )
        if not os.path.exists(model_path):
            print(f"\n{pipeline}: missing {model_path}")
            continue

        model = WorldModelV4(V4Config())
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()

        metadata = (
            torch.load(checkpoint_path, map_location="cpu")
            if os.path.exists(checkpoint_path)
            else {}
        )
        threshold = metadata.get("adaptive_threshold")
        if threshold is None:
            threshold = metadata.get("validation_p90_drift")

        generator = RolloutGeneratorV4(
            model,
            strategy=pipeline,
            fixed_interval=int(metadata.get("fixed_interval", 8)),
            adaptive_threshold=threshold,
            runtime=runtime,
        )
        prediction = generator.generate(
            initial,
            actions,
            real_sequence=real if pipeline != "baseline" else None,
        ).cpu()

        pred_motion = torch.abs(
            prediction[1:] - prediction[:-1]
        ).mean(dim=(1, 2, 3)).numpy()
        real_motion = torch.abs(
            real[1:] - real[:-1]
        ).mean(dim=(1, 2, 3)).numpy()

        ghost_score, ghost_active_fraction = departed_pixel_ghost_score(
            real,
            prediction,
        )

        action_outputs = []
        with torch.no_grad():
            for action in range(4):
                hidden = model.init_hidden(1, device)
                output, _ = model(
                    initial,
                    torch.tensor([action], device=device),
                    hidden,
                )
                action_outputs.append(output)
        runtime.sync()
        action_outputs = [output.cpu() for output in action_outputs]

        action_diffs = []
        for i in range(4):
            for j in range(i + 1, 4):
                action_diffs.append(
                    float(torch.abs(action_outputs[i] - action_outputs[j]).mean())
                )

        print(f"\n===== {pipeline.upper()} =====")
        print("Mean REAL motion:", float(real_motion.mean()))
        print("Mean PREDICTED motion:", float(pred_motion.mean()))
        print(
            "Motion ratio:",
            float(pred_motion.mean() / max(real_motion.mean(), 1e-12)),
        )
        print("First 20 predicted motion:", np.round(pred_motion[:20], 7))
        print("Mean action-conditioned difference:", float(np.mean(action_diffs)))
        print("Departed-pixel ghost brightness:", ghost_score)
        print("Departed pixels >0.10 brightness:", ghost_active_fraction)
        print("Corrections:", len(generator.correction_steps))
        if threshold is not None:
            print("Adaptive/calibrated threshold:", threshold)


if __name__ == "__main__":
    main()
