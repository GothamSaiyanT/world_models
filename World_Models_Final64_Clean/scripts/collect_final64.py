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

from utils.experiment_logging import (
    environment_info,
    utc_now_iso,
)


gym.register_envs(ale_py)


def preprocess(frame):
    """
    Convert an Atari RGB observation into the 64x64
    grayscale representation used by the world model.
    """

    frame = frame[CROP_TOP:CROP_BOTTOM]

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY,
    )

    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return gray.astype(np.uint8)


def start_episode(env):
    """
    Reset the environment and issue FIRE so that
    Breakout begins in an active state.

    Returns the first active observation.
    """

    observation, _ = env.reset()

    observation, _, terminated, truncated, _ = env.step(1)

    # Extremely defensive fallback.
    if terminated or truncated:
        observation, _ = env.reset()
        observation, _, _, _, _ = env.step(1)

    return observation


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-frames",
        type=int,
        default=NUM_FRAMES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_FOLDER),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    output = args.output

    os.makedirs(
        output,
        exist_ok=True,
    )

    frames_path = os.path.join(
        output,
        "frames.npy",
    )

    actions_path = os.path.join(
        output,
        "actions.npy",
    )

    episode_ids_path = os.path.join(
        output,
        "episode_ids.npy",
    )

    existing_files = [
        frames_path,
        actions_path,
        episode_ids_path,
    ]

    if (
        any(os.path.exists(path) for path in existing_files)
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Dataset already exists in {output}. "
            "Use --overwrite only if you intentionally "
            "want to replace it."
        )

    # ---------------------------------------------------------
    # Reproducible action generator
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    env = gym.make(
        ENVIRONMENT,
        render_mode="rgb_array",
    )

    observation, info = env.reset(
        seed=args.seed
    )

    meanings = tuple(
        env.unwrapped.get_action_meanings()
    )

    if meanings[:4] != EXPECTED_ACTION_MEANINGS:

        env.close()

        raise RuntimeError(
            f"Unexpected action meanings: {meanings}. "
            f"Expected first four: "
            f"{EXPECTED_ACTION_MEANINGS}."
        )

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    frames = []
    actions = []
    episode_ids = []

    episode_resets = 0
    episode_id = 0

    # ---------------------------------------------------------
    # Start first active Breakout episode
    # ---------------------------------------------------------

    observation, _, terminated, truncated, _ = env.step(1)

    if terminated or truncated:

        observation, _ = env.reset(
            seed=args.seed
        )

        observation, _, _, _, _ = env.step(1)

    # ---------------------------------------------------------
    # Collection
    # ---------------------------------------------------------

    for step in range(args.num_frames):

        # -----------------------------------------------------
        # Save CURRENT observation
        # -----------------------------------------------------

        frames.append(
            preprocess(observation)
        )

        episode_ids.append(
            episode_id
        )

        # -----------------------------------------------------
        # Choose action
        # -----------------------------------------------------

        action = int(
            rng.choice(
                np.arange(4),
                p=np.asarray(
                    ACTION_PROBABILITIES,
                    dtype=np.float64,
                ),
            )
        )

        actions.append(
            action
        )

        # -----------------------------------------------------
        # Execute action
        # -----------------------------------------------------

        next_observation, _, terminated, truncated, _ = env.step(
            action
        )

        # -----------------------------------------------------
        # Handle episode termination
        # -----------------------------------------------------

        if terminated or truncated:

            episode_resets += 1
            episode_id += 1

            observation, _ = env.reset()

            # FIRE to start the next Breakout episode.
            observation, _, fire_terminated, fire_truncated, _ = env.step(1)

            if fire_terminated or fire_truncated:
                observation, _ = env.reset()
                observation, _, _, _, _ = env.step(1)

        else:

            observation = next_observation

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if (
            (step + 1) % 1000 == 0
            or step + 1 == args.num_frames
        ):

            print(
                f"Collected "
                f"{step + 1:,} / "
                f"{args.num_frames:,} frames",
                flush=True,
            )

    env.close()

    # ---------------------------------------------------------
    # Convert to arrays
    # ---------------------------------------------------------

    frames = np.asarray(
        frames,
        dtype=np.uint8,
    )

    actions = np.asarray(
        actions,
        dtype=np.int64,
    )

    episode_ids = np.asarray(
        episode_ids,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    np.save(
        frames_path,
        frames,
    )

    np.save(
        actions_path,
        actions,
    )

    np.save(
        episode_ids_path,
        episode_ids,
    )

    # ---------------------------------------------------------
    # Motion statistics
    #
    # Only calculate motion for consecutive frames belonging
    # to the same episode. Episode resets are not real motion.
    # ---------------------------------------------------------

    normalized = (
        frames.astype(np.float32)
        / 255.0
    )

    same_episode = (
        episode_ids[1:]
        == episode_ids[:-1]
    )

    consecutive_motion = np.mean(
        np.abs(
            normalized[1:]
            - normalized[:-1]
        ),
        axis=(1, 2),
    )

    valid_motion = consecutive_motion[
        same_episode
    ]

    # ---------------------------------------------------------
    # Action statistics
    # ---------------------------------------------------------

    unique, counts = np.unique(
        actions,
        return_counts=True,
    )

    action_counts = {
        str(int(action)): int(count)
        for action, count
        in zip(unique, counts)
    }

    # ---------------------------------------------------------
    # Episode statistics
    # ---------------------------------------------------------

    unique_episodes, episode_counts = np.unique(
        episode_ids,
        return_counts=True,
    )

    episode_lengths = {
        str(int(ep)): int(count)
        for ep, count
        in zip(
            unique_episodes,
            episode_counts,
        )
    }

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {

        "created_utc":
            utc_now_iso(),

        "environment":
            ENVIRONMENT,

        "seed":
            args.seed,

        "num_frames":
            int(len(frames)),

        "frame_shape":
            list(frames.shape),

        "frame_dtype":
            str(frames.dtype),

        "frame_min":
            int(frames.min()),

        "frame_max":
            int(frames.max()),

        "preprocessing": {

            "crop_rows": [
                CROP_TOP,
                CROP_BOTTOM,
            ],

            "grayscale":
                True,

            "resize": [
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],

            "stored_as":
                "uint8",

            "training_normalization":
                "divide by 255.0 on load",
        },

        "action_meanings":
            list(meanings),

        "action_probabilities":
            list(ACTION_PROBABILITIES),

        "action_counts":
            action_counts,

        "episode_resets":
            int(episode_resets),

        "episode_count":
            int(
                len(unique_episodes)
            ),

        "episode_lengths":
            episode_lengths,

        "motion": {

            "mean":
                float(valid_motion.mean())
                if len(valid_motion)
                else 0.0,

            "median":
                float(
                    np.median(valid_motion)
                )
                if len(valid_motion)
                else 0.0,

            "p90":
                float(
                    np.percentile(
                        valid_motion,
                        90,
                    )
                )
                if len(valid_motion)
                else 0.0,

            "max":
                float(valid_motion.max())
                if len(valid_motion)
                else 0.0,
        },

        "environment_info":
            environment_info(
                os.getcwd()
            ),
    }

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata_path = os.path.join(
        output,
        "dataset_metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET SAVED")
    print("=" * 60)

    print(
        "Frames:",
        frames.shape,
        frames.dtype,
        "range",
        frames.min(),
        frames.max(),
    )

    print(
        "Actions:",
        actions.shape,
    )

    print(
        "Episode IDs:",
        episode_ids.shape,
    )

    print(
        "Episodes:",
        len(unique_episodes),
    )

    print(
        "Episode resets:",
        episode_resets,
    )

    print(
        "Mean normalized motion:",
        metadata["motion"]["mean"],
    )

    print(
        "Action counts:",
        action_counts,
    )

    print(
        "Folder:",
        output,
    )

    print("=" * 60)


if __name__ == "__main__":
    main()