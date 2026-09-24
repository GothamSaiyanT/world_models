import argparse

from training.data_collector_v3 import BreakoutCollectorV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--folder", default="data_v3")
    args = parser.parse_args()
    BreakoutCollectorV3(save_folder=args.folder).collect(args.steps)


if __name__ == "__main__":
    main()
