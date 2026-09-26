from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data_v3")
    parser.add_argument("--output", default="data_v4_64")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    frames = np.load(source / "frames.npy", mmap_mode="r")
    actions = np.load(source / "actions.npy")

    if frames.ndim != 3:
        raise ValueError(f"Expected (N,H,W) frames; got {frames.shape}")
    if len(actions) != len(frames):
        raise ValueError("Actions must have the same length as frames.")

    print("Source frames:", frames.shape, frames.dtype)
    print("Actions:", actions.shape)

    target_path = output / "frames.npy"
    target = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frames), 64, 64),
    )

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(frames), batch_size):
        end = min(start + batch_size, len(frames))
        batch_np = np.asarray(frames[start:end], dtype=np.float32)
        batch = torch.from_numpy(batch_np).unsqueeze(1)
        resized = F.interpolate(
            batch,
            size=(64, 64),
            mode="area",
        ).squeeze(1)
        target[start:end] = resized.numpy()
        if start == 0 or end == len(frames) or start % (batch_size * 10) == 0:
            print(f"Converted {end}/{len(frames)} frames")

    del target
    np.save(output / "actions.npy", actions)

    check = np.load(target_path, mmap_mode="r")
    motion = np.abs(check[1:] - check[:-1]).mean(axis=(1, 2))
    print("V4 frames:", check.shape, check.dtype)
    print("Mean motion:", float(motion.mean()))
    print("Median motion:", float(np.median(motion)))
    print("P90 motion:", float(np.percentile(motion, 90)))
    print("Saved to:", output)


if __name__ == "__main__":
    main()
