"""Sanity-check the ball-detection heuristic before a full training run.

detect_ball_mask (core/loss.py) isolates the ball by connected-component
SIZE (a small isolated bright blob), not by its position in the frame. An
earlier version tried to exclude the paddle/walls by position instead, and
that broke silently on this dataset: the brick rows near the top are bright
too, so a position-based exclusion caught the entire brick block as "ball"
(hundreds of pixels), swamping the actual ~4-pixel ball. The size-based
version doesn't care where the ball is, only how big the bright blob is —
but "how big is the ball at your resolution/brightness threshold" is still
a guess worth checking. This script overlays what the heuristic currently
flags on real frames so you can confirm it's catching the ball and nothing
bigger.

Usage:
    python scripts/inspect_ball_mask.py --frames 0 25 50 75
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config_final64 import (
    BALL_BRIGHTNESS_THRESHOLD,
    BALL_MAX_AREA_PX,
    BALL_MIN_AREA_PX,
    DATA_FOLDER,
    OUTPUT_FOLDER,
)
from core.loss import detect_ball_mask
from training.final64_dataset import load_normalized_frames


def render_overlay(frame, brightness_threshold, min_area, max_area, scale=6):
    height, width = frame.shape

    frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    mask = detect_ball_mask(
        frame_t,
        brightness_threshold=brightness_threshold,
        min_ball_area=min_area,
        max_ball_area=max_area,
    ).squeeze(0).squeeze(0).numpy()

    rgb = np.stack([frame, frame, frame], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    # Paint what the heuristic flags as "ball" in red.
    rgb[mask] = [1.0, 0.15, 0.15]

    image = Image.fromarray((rgb * 255.0).astype(np.uint8))
    image = image.resize((width * scale, height * scale), Image.NEAREST)
    return image, int(mask.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 25, 50, 75])
    parser.add_argument("--brightness-threshold", type=float, default=BALL_BRIGHTNESS_THRESHOLD)
    parser.add_argument("--min-area", type=int, default=BALL_MIN_AREA_PX)
    parser.add_argument("--max-area", type=int, default=BALL_MAX_AREA_PX)
    args = parser.parse_args()

    frames = load_normalized_frames(DATA_FOLDER)

    output_dir = Path(OUTPUT_FOLDER) / "ball_mask_checks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in args.frames:
        if frame_index < 0 or frame_index >= len(frames):
            print(f"Skipping {frame_index}: out of range (0..{len(frames) - 1})")
            continue

        overlay, flagged_pixels = render_overlay(
            frames[frame_index],
            args.brightness_threshold,
            args.min_area,
            args.max_area,
        )
        out_path = output_dir / f"frame_{frame_index}_mask.png"
        overlay.save(out_path)
        print(f"Saved: {out_path} ({flagged_pixels} pixels flagged)")

    print(
        "\nCheck each saved image: red pixels are what the loss will treat as "
        "'the ball'. Nothing red, on a frame where the ball is visible, means "
        "lower --brightness-threshold or raise --max-area. A big red blob on "
        "the paddle or a brick means --max-area is too high — lower it until "
        "only the ball-sized blob survives."
    )


if __name__ == "__main__":
    main()
