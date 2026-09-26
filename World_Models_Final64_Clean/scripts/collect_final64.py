import argparse
import json
import os

import cv2
import gymnasium as gym
import ale_py
import numpy as np

from config_final64 import (
    ACTION_PROBABILITIES,
    CROP_BOTTOM,
    CROP_TOP,
    DATA_FOLDER,
    ENVIRONMENT,
    EXPECTED_ACTION_MEANINGS,
    IMAGE_SIZE,
    NUM_FRAMES,
    SEED,
)
from utils.experiment_logging import environment_info, utc_now_iso


gym.register_envs(ale_py)


def preprocess(frame):
    frame = frame[CROP_TOP:CROP_BOTTOM]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return gray.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=str(DATA_FOLDER))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output
    os.makedirs(output, exist_ok=True)

    frames_path = os.path.join(output, "frames.npy")
    actions_path = os.path.join(output, "actions.npy")

    if (os.path.exists(frames_path) or os.path.exists(actions_path)) and not args.overwrite:
        raise FileExistsError(
            f"Dataset already exists in {output}. Use --overwrite only if you intentionally want to replace it."
        )

    rng = np.random.default_rng(args.seed)

    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    observation, info = env.reset(seed=args.seed)

    meanings = tuple(env.unwrapped.get_action_meanings())
    if meanings[:4] != EXPECTED_ACTION_MEANINGS:
        env.close()
        raise RuntimeError(
            f"Unexpected action meanings: {meanings}. Expected first four: {EXPECTED_ACTION_MEANINGS}."
        )

    frames = []
    actions = []
    episode_resets = 0

    # Start the first Breakout episode.
    observation, _, terminated, truncated, _ = env.step(1)
    if terminated or truncated:
        observation, info = env.reset(seed=args.seed)

    for step in range(args.num_frames):
        frames.append(preprocess(observation))

        action = int(
            rng.choice(
                np.arange(4),
                p=np.asarray(ACTION_PROBABILITIES, dtype=np.float64),
            )
        )
        actions.append(action)

        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            episode_resets += 1
            observation, info = env.reset()
            observation, _, terminated, truncated, _ = env.step(1)

        if (step + 1) % 1000 == 0 or step + 1 == args.num_frames:
            print(f"Collected {step + 1:,} / {args.num_frames:,} frames", flush=True)

    env.close()

    frames = np.asarray(frames, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.int64)

    np.save(frames_path, frames)
    np.save(actions_path, actions)

    normalized = frames.astype(np.float32) / 255.0
    motion = np.mean(
        np.abs(normalized[1:] - normalized[:-1]),
        axis=(1, 2),
    )

    unique, counts = np.unique(actions, return_counts=True)
    action_counts = {str(int(a)): int(c) for a, c in zip(unique, counts)}

    metadata = {
        "created_utc": utc_now_iso(),
        "environment": ENVIRONMENT,
        "seed": args.seed,
        "num_frames": int(len(frames)),
        "frame_shape": list(frames.shape),
        "frame_dtype": str(frames.dtype),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "preprocessing": {
            "crop_rows": [CROP_TOP, CROP_BOTTOM],
            "grayscale": True,
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "stored_as": "uint8",
            "training_normalization": "divide by 255.0 on load",
        },
        "action_meanings": list(meanings),
        "action_probabilities": list(ACTION_PROBABILITIES),
        "action_counts": action_counts,
        "episode_resets": int(episode_resets),
        "motion": {
            "mean": float(motion.mean()),
            "median": float(np.median(motion)),
            "p90": float(np.percentile(motion, 90)),
            "max": float(motion.max()),
        },
        "environment_info": environment_info(os.getcwd()),
    }

    with open(os.path.join(output, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset saved")
    print("Frames:", frames.shape, frames.dtype, "range", frames.min(), frames.max())
    print("Actions:", actions.shape)
    print("Mean normalized motion:", metadata["motion"]["mean"])
    print("Action counts:", action_counts)
    print("Folder:", output)


if __name__ == "__main__":
    main()
