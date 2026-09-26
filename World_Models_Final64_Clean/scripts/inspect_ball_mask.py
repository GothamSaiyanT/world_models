"""Sanity-check the ball-detection heuristic before a full training run.

detect_ball_mask (core/loss.py) assumes a specific frame layout: a bright
wall border a few pixels wide, and the paddle confined to a band of rows
near the bottom. Those assumptions are guesses about your cropped dataset's
geometry — this script overlays the detected mask on a handful of real
frames so you can eyeball whether it's actually isolating the ball, or
whether it's leaking onto the paddle/walls (or missing the ball entirely).

Usage:
    python scripts/inspect_ball_mask.py --frames 0 25 50 75
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from config_final64 import (
    BALL_BRIGHTNESS_THRESHOLD,
    BALL_PADDLE_BAND_PX,
    BALL_WALL_MARGIN_PX,
    DATA_FOLDER,
    OUTPUT_FOLDER,
)
from training.final64_dataset import load_normalized_frames


def render_overlay(frame, wall_margin, paddle_band, brightness_threshold, scale=6):
    height, width = frame.shape

    row_end = max(height - paddle_band, wall_margin + 1)
    col_end = max(width - wall_margin, wall_margin + 1)

    mask = np.zeros_like(frame, dtype=bool)
    interior = frame[wall_margin:row_end, wall_margin:col_end]
    mask[wall_margin:row_end, wall_margin:col_end] = interior > brightness_threshold

    rgb = np.stack([frame, frame, frame], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    # Paint detected ball pixels red so they stand out against the greyscale frame.
    rgb[mask] = [1.0, 0.15, 0.15]

    image = Image.fromarray((rgb * 255.0).astype(np.uint8))
    image = image.resize((width * scale, height * scale), Image.NEAREST)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 25, 50, 75])
    parser.add_argument("--wall-margin", type=int, default=BALL_WALL_MARGIN_PX)
    parser.add_argument("--paddle-band", type=int, default=BALL_PADDLE_BAND_PX)
    parser.add_argument("--brightness-threshold", type=float, default=BALL_BRIGHTNESS_THRESHOLD)
    args = parser.parse_args()

    frames = load_normalized_frames(DATA_FOLDER)

    output_dir = Path(OUTPUT_FOLDER) / "ball_mask_checks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in args.frames:
        if frame_index < 0 or frame_index >= len(frames):
            print(f"Skipping {frame_index}: out of range (0..{len(frames) - 1})")
            continue

        overlay = render_overlay(
            frames[frame_index],
            args.wall_margin,
            args.paddle_band,
            args.brightness_threshold,
        )
        out_path = output_dir / f"frame_{frame_index}_mask.png"
        overlay.save(out_path)
        print("Saved:", out_path)

    print(
        "\nCheck each saved image: red pixels are what the loss will treat as "
        "'the ball'. If red also covers the paddle or the wall border, "
        "increase --paddle-band / --wall-margin. If nothing is red on a frame "
        "where the ball is visible, lower --brightness-threshold."
    )


if __name__ == "__main__":
    main()
