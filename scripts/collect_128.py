"""Collect fresh native 128x128 Breakout frames for high-resolution training."""

import argparse
import os
import shutil

from highres_128 import DATA_FOLDER, IMAGE_SIZE, validate_128_dataset
from training.data_collector import DataCollector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--folder", default=DATA_FOLDER)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing target folder before collecting fresh data.",
    )
    args = parser.parse_args()

    if args.steps < 100:
        raise ValueError("Use at least 100 steps for data collection.")

    if args.force and os.path.isdir(args.folder):
        shutil.rmtree(args.folder)

    os.makedirs(args.folder, exist_ok=True)

    collector = DataCollector(
        image_size=IMAGE_SIZE,
        save_folder=args.folder,
    )
    collector.collect(num_steps=args.steps)

    frame_shape, action_shape = validate_128_dataset(args.folder)
    print("128x128 dataset validated.")
    print("Frames:", frame_shape)
    print("Actions:", action_shape)


if __name__ == "__main__":
    main()
