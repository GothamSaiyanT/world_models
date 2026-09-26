import shutil
from pathlib import Path

from config_final64 import DATA_FOLDER, PAPER_RESULTS_FOLDER, RESULT_FOLDER


RESULT_FILES = [
    "final_multistart_summary.csv",
    "final_multistart_windows.csv",
    "final_multistart_frames.csv",
    "final_experiment_metadata.json",
    "final_research_summary.md",
    "training_metadata.json",
    "training_history.csv",
    "training_curve.png",
    "dataset_validation.json",
    "dataset_preview.png",
]

DATA_FILES = [
    "dataset_metadata.json",
]


def copy_if_present(source, target, name, copied, missing):
    src = source / name
    if src.exists():
        shutil.copy2(src, target / name)
        copied.append(name)
    else:
        missing.append(str(src))


def main():
    result_source = Path(RESULT_FOLDER)
    data_source = Path(DATA_FOLDER)
    target = Path(PAPER_RESULTS_FOLDER)
    target.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for name in RESULT_FILES:
        copy_if_present(result_source, target, name, copied, missing)

    for name in DATA_FILES:
        copy_if_present(data_source, target, name, copied, missing)

    print("Copied paper-safe artifacts:")
    for name in copied:
        print(" -", name)

    if missing:
        print("\nNot present yet:")
        for name in missing:
            print(" -", name)

    print("\nThese small files can be committed after the final experiment if you want the numerical record in Git.")


if __name__ == "__main__":
    main()
