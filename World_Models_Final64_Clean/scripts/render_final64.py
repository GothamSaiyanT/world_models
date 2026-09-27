import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config_final64 import (
    ADAPTIVE_THRESHOLD,
    DATA_FOLDER,
    EVALUATION_HORIZON,
    FIXED_INTERVAL,
    HIDDEN_SIZE,
    IMAGE_SIZE,
    LATENT_SIZE,
    MODEL_PATH,
    OUTPUT_FOLDER,
)
from core.world_model import WorldModel
from training.final64_dataset import load_actions, load_normalized_frames
from utils.rollout_final64 import generate_rollout


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


def to_panel(frame, title, frame_number, correction=False, scale=4):
    image = np.clip(frame, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    image = cv2.resize(
        image,
        (IMAGE_SIZE * scale, IMAGE_SIZE * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    top = 42
    canvas = np.zeros((image.shape[0] + top, image.shape[1], 3), dtype=np.uint8)
    canvas[top:] = image

    cv2.putText(
        canvas,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"frame {frame_number}",
        (8, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    if correction:
        cv2.putText(
            canvas,
            "CORRECTION",
            (image.shape[1] - 92, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=EVALUATION_HORIZON)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--fixed-interval", type=int, default=FIXED_INTERVAL)
    parser.add_argument("--adaptive-threshold", type=float, default=ADAPTIVE_THRESHOLD)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    frames = load_normalized_frames(DATA_FOLDER)
    actions = load_actions(DATA_FOLDER)

    if args.start < 0 or args.start + args.horizon >= len(frames):
        raise ValueError("Requested start/horizon exceeds the dataset.")

    model = WorldModel(
        latent_size=LATENT_SIZE,
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
    )
    epoch, best_loss = model.load(str(MODEL_PATH))
    model.to(device)
    model.eval()

    real = frames[args.start:args.start + args.horizon + 1]
    action_sequence = actions[args.start:args.start + args.horizon]

    results = {}
    for strategy in ("baseline", "fixed", "adaptive"):
        results[strategy] = generate_rollout(
            model=model,
            real_frames=real,
            actions=action_sequence,
            strategy=strategy,
            fixed_interval=args.fixed_interval,
            adaptive_threshold=args.adaptive_threshold,
            device=device,
        )

    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"final64_start_{args.start}_h{args.horizon}.mp4"

    scale = 4
    sample_panel = to_panel(real[0], "Ground Truth", 0, scale=scale)
    panel_h, panel_w = sample_panel.shape[:2]
    video_size = (panel_w * 2, panel_h * 2)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        video_size,
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer. Try installing opencv-python-headless again.")

    for frame_index in range(args.horizon + 1):
        gt = to_panel(real[frame_index], "Ground Truth", frame_index, scale=scale)
        baseline = to_panel(
            results["baseline"]["prediction"][frame_index],
            "Baseline",
            frame_index,
            scale=scale,
        )
        fixed = to_panel(
            results["fixed"]["prediction"][frame_index],
            "Fixed Interval",
            frame_index,
            correction=(frame_index in results["fixed"]["corrections"]),
            scale=scale,
        )
        adaptive = to_panel(
            results["adaptive"]["prediction"][frame_index],
            "Adaptive",
            frame_index,
            correction=(frame_index in results["adaptive"]["corrections"]),
            scale=scale,
        )

        top = np.hstack([gt, baseline])
        bottom = np.hstack([fixed, adaptive])
        frame = np.vstack([top, bottom])
        writer.write(frame)

    writer.release()

    print("Loaded epoch:", epoch)
    print("Best validation loss:", best_loss)
    print("Fixed effective corrections:", results["fixed"]["corrections"])
    print("Adaptive effective corrections:", results["adaptive"]["corrections"])
    print("Saved video:", output_path)


if __name__ == "__main__":
    main()
