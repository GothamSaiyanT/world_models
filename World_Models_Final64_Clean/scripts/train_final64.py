import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config_final64 import (
    BALL_BRIGHTNESS_THRESHOLD,
    BALL_PADDLE_BAND_PX,
    BALL_WALL_MARGIN_PX,
    BALL_WEIGHT,
    BATCH_SIZE,
    DATA_FOLDER,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_SIZE,
    IMAGE_SIZE,
    LATENT_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_FOLDER,
    MODEL_PATH,
    MOTION_THRESHOLD,
    MOTION_WEIGHT,
    RESULT_FOLDER,
    SEED,
    SEQUENCE_LENGTH,
    TRAIN_END,
)
from core.loss import BallAwareMotionWeightedMSELoss
from core.world_model import WorldModel
from training.final64_dataset import SequenceRangeDataset
from training.optimizer import Adam
from training.trainer import clip_gradient_norm
from utils.experiment_logging import environment_info, write_json


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batches = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for frames, actions in loader:
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
            actions = actions.to(device=device, dtype=torch.long, non_blocking=True)

            batch_size = frames.shape[0]
            hidden = model.init_hidden(batch_size=batch_size, device=device)

            if training:
                optimizer.zero_grad()

            sequence_loss = torch.zeros((), device=device, dtype=torch.float32)

            for timestep in range(actions.shape[1]):
                current_frame = frames[:, timestep]
                target_frame = frames[:, timestep + 1]
                current_action = actions[:, timestep]

                prediction, hidden = model(current_frame, current_action, hidden)

                sequence_loss = sequence_loss + criterion(
                    prediction=prediction,
                    target=target_frame,
                    current_frame=current_frame,
                )

            sequence_loss = sequence_loss / actions.shape[1]

            if training:
                sequence_loss.backward()
                clip_gradient_norm(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(sequence_loss.item())
            batches += 1

    if batches == 0:
        raise RuntimeError("DataLoader produced no batches.")

    return total_loss / batches


def save_history(history, result_folder):
    result_folder = Path(result_folder)
    result_folder.mkdir(parents=True, exist_ok=True)

    csv_path = result_folder / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best"])
        writer.writeheader()
        writer.writerows(history)

    json_path = result_folder / "training_history.json"
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Motion-weighted MSE")
    plt.title("Final64 training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_folder / "training_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)

    data_folder = Path(DATA_FOLDER)
    model_folder = Path(MODEL_FOLDER)
    result_folder = Path(RESULT_FOLDER)
    model_folder.mkdir(parents=True, exist_ok=True)
    result_folder.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"{MODEL_PATH} already exists. Use --overwrite only for an intentional fresh retraining."
        )

    frame_count = len(np.load(data_folder / "frames.npy", mmap_mode="r"))

    train_dataset = SequenceRangeDataset(
        data_folder,
        start=0,
        end=TRAIN_END,
        sequence_length=SEQUENCE_LENGTH,
    )
    val_dataset = SequenceRangeDataset(
        data_folder,
        start=TRAIN_END,
        end=frame_count,
        sequence_length=SEQUENCE_LENGTH,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = WorldModel(
        latent_size=LATENT_SIZE,
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
    ).to(device)

    criterion = BallAwareMotionWeightedMSELoss(
        motion_weight=MOTION_WEIGHT,
        motion_threshold=MOTION_THRESHOLD,
        ball_weight=BALL_WEIGHT,
        wall_margin=BALL_WALL_MARGIN_PX,
        paddle_band=BALL_PADDLE_BAND_PX,
        ball_brightness_threshold=BALL_BRIGHTNESS_THRESHOLD,
    )
    optimizer = Adam(list(model.parameters()), args.learning_rate)

    parameter_count = sum(int(p.data.numel()) for p in model.parameters())

    print("Total frames:", frame_count)
    print("Train frames:", TRAIN_END)
    print("Validation frames:", frame_count - TRAIN_END)
    print("Model parameters:", f"{parameter_count:,}")
    print("Accelerator:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("Torch device:", device)
    print("Batch size:", args.batch_size)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Learning rate:", args.learning_rate)
    print("Early stopping patience:", args.patience)

    best_val = float("inf")
    best_epoch = 0
    no_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss = run_epoch(model, val_loader, criterion, None, device, training=False)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improvement = 0
            model.save(str(MODEL_PATH), epoch=epoch, best_loss=best_val)
        else:
            no_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best": bool(improved),
            }
        )
        save_history(history, result_folder)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )
        if improved:
            print("  Best checkpoint updated.")
        else:
            print(f"  No validation improvement: {no_improvement}/{args.patience}")

        if no_improvement >= args.patience:
            print("Early stopping.")
            break

    elapsed = time.perf_counter() - started

    metadata = {
        "model": {
            "checkpoint": str(MODEL_PATH),
            "parameter_count": parameter_count,
            "latent_size": LATENT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "image_size": IMAGE_SIZE,
        },
        "training": {
            "seed": args.seed,
            "train_end": TRAIN_END,
            "validation_start": TRAIN_END,
            "frame_count": frame_count,
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "motion_weight": MOTION_WEIGHT,
            "motion_threshold": MOTION_THRESHOLD,
            "ball_weight": BALL_WEIGHT,
            "ball_wall_margin_px": BALL_WALL_MARGIN_PX,
            "ball_paddle_band_px": BALL_PADDLE_BAND_PX,
            "ball_brightness_threshold": BALL_BRIGHTNESS_THRESHOLD,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val,
            "elapsed_seconds": elapsed,
        },
        "environment": environment_info(Path(__file__).resolve().parents[1]),
    }
    write_json(result_folder / "training_metadata.json", metadata)

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val)
    print("Model:", MODEL_PATH)
    print("History:", result_folder / "training_history.csv")


if __name__ == "__main__":
    main()
