"""Frozen configuration for the final 64x64 Breakout world-model experiment.

The goal is reproducibility: training, evaluation, dashboard and rendering all import
these defaults so the paper and demo use the same settings.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FOLDER = PROJECT_ROOT / "data_final64"
MODEL_FOLDER = PROJECT_ROOT / "models_final64"
RESULT_FOLDER = PROJECT_ROOT / "results_final64"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs_final64"
PAPER_RESULTS_FOLDER = PROJECT_ROOT / "paper_results"

ENVIRONMENT = "ALE/Breakout-v5"
SEED = 42
NUM_FRAMES = 10_000
IMAGE_SIZE = 64

# The final dataset preprocessing used in the successful experiment.
CROP_TOP = 30
CROP_BOTTOM = 200
ACTION_PROBABILITIES = (0.10, 0.10, 0.40, 0.40)
EXPECTED_ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")

# Shared learned predictor.
LATENT_SIZE = 128
HIDDEN_SIZE = 128
SEQUENCE_LENGTH = 16
TRAIN_END = 8_000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
MOTION_WEIGHT = 2.0
MOTION_THRESHOLD = 0.05

# Frozen final evaluation protocol.
EVALUATION_STARTS = (0, 500, 1000, 2000, 3000)
EVALUATION_HORIZON = 100
FIXED_INTERVAL = 10
ADAPTIVE_THRESHOLD = 0.0005

MODEL_PATH = MODEL_FOLDER / "best_world_model.npz"
TRAINING_HISTORY_CSV = RESULT_FOLDER / "training_history.csv"
TRAINING_METADATA_JSON = RESULT_FOLDER / "training_metadata.json"
DATASET_METADATA_JSON = DATA_FOLDER / "dataset_metadata.json"
FINAL_WINDOWS_CSV = RESULT_FOLDER / "final_multistart_windows.csv"
FINAL_FRAMES_CSV = RESULT_FOLDER / "final_multistart_frames.csv"
FINAL_SUMMARY_CSV = RESULT_FOLDER / "final_multistart_summary.csv"
FINAL_METADATA_JSON = RESULT_FOLDER / "final_experiment_metadata.json"
FINAL_REPORT_MD = RESULT_FOLDER / "final_research_summary.md"
