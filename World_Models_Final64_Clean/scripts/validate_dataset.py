import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from config_final64 import DATA_FOLDER, RESULT_FOLDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_FOLDER))
    parser.add_argument("--output", default=str(RESULT_FOLDER))
    args = parser.parse_args()

    frames = np.load(os.path.join(args.data, "frames.npy"))
    actions = np.load(os.path.join(args.data, "actions.npy"))

    x = frames.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0

    motion = np.mean(np.abs(x[1:] - x[:-1]), axis=(1, 2))
    unique, counts = np.unique(actions, return_counts=True)

    stats = {
        "frames_shape": list(frames.shape),
        "frames_dtype": str(frames.dtype),
        "frames_min": float(frames.min()),
        "frames_max": float(frames.max()),
        "actions_shape": list(actions.shape),
        "action_counts": {str(int(a)): int(c) for a, c in zip(unique, counts)},
        "mean_motion": float(motion.mean()),
        "median_motion": float(np.median(motion)),
        "p90_motion": float(np.percentile(motion, 90)),
        "max_motion": float(motion.max()),
    }

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "dataset_validation.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    indices = [0, 500, 1000, 3000, 5000, 8000]
    indices = [i for i in indices if i < len(frames)]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, indices):
        ax.imshow(x[idx], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    plt.tight_layout()
    preview = os.path.join(args.output, "dataset_preview.png")
    plt.savefig(preview, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("===== DATASET VALIDATION =====")
    print("Frames:", frames.shape, frames.dtype)
    print("Range:", frames.min(), frames.max())
    print("Mean motion:", stats["mean_motion"])
    print("Median motion:", stats["median_motion"])
    print("P90 motion:", stats["p90_motion"])
    for action, count in zip(unique, counts):
        print(f"Action {int(action)}: {int(count)} ({100.0 * count / len(actions):.2f}%)")
    print("Preview:", preview)


if __name__ == "__main__":
    main()
