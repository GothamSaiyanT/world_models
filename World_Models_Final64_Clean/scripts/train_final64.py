import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config_final64 import (
    BATCH_SIZE,
    DATA_FOLDER,
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

from core.loss import StableMotionWeightedMSELoss
from core.world_model import WorldModel
from training.final64_dataset import SequenceRangeDataset
from training.optimizer import Adam
from training.trainer import clip_gradient_norm
from utils.experiment_logging import environment_info, write_json


# ============================================================
# DEVICE
# ============================================================

def choose_device(requested):

    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but not available."
        )

    return torch.device(requested)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
):
    """
    Run one complete training or validation epoch.

    FINAL BETA TRAINING STRATEGY:
    --------------------------------
    Teacher-forced sequential training.

    At every timestep the model receives the REAL current frame.

        real frame[t]
              |
              v
           Encoder
              |
              v
           Dynamics <--- previous hidden state
              |
              v
           Decoder
              |
              v
        prediction[t+1]

    The important part is that the GRU hidden state is NOT reset
    between timesteps.

    Therefore the model still learns temporal information across
    the complete sequence while avoiding prediction artifacts being
    fed back into the encoder during training.
    """

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batches = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for frames, actions in loader:

            # ----------------------------------------------------
            # Move batch to selected device
            # ----------------------------------------------------

            frames = frames.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            actions = actions.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            batch_size = frames.shape[0]

            # ----------------------------------------------------
            # Initialise recurrent hidden state
            #
            # IMPORTANT:
            # This happens ONCE per sequence/batch.
            # It is NOT reset inside the timestep loop.
            # ----------------------------------------------------

            hidden = model.init_hidden(
                batch_size=batch_size,
                device=device,
            )

            if training:
                optimizer.zero_grad()

            sequence_loss = torch.zeros(
                (),
                device=device,
                dtype=torch.float32,
            )

            # ----------------------------------------------------
            # TEACHER-FORCED SEQUENTIAL TRAINING
            # ----------------------------------------------------

            for timestep in range(
                actions.shape[1]
            ):

                # Real frame at the current timestep.
                current_frame = frames[
                    :,
                    timestep
                ]

                # Real frame immediately after the action.
                target_frame = frames[
                    :,
                    timestep + 1
                ]

                # Action connecting:
                #
                # current_frame -> target_frame
                #
                current_action = actions[
                    :,
                    timestep
                ]

                # ------------------------------------------------
                # Predict next frame.
                #
                # The image input is the REAL current frame.
                #
                # However, hidden comes from the previous
                # timestep and therefore carries temporal
                # information through the sequence.
                # ------------------------------------------------

                prediction, hidden = model(
                    current_frame,
                    current_action,
                    hidden,
                )

                # ------------------------------------------------
                # Motion-weighted reconstruction loss
                # ------------------------------------------------

                step_loss = criterion(
                    prediction=prediction,
                    target=target_frame,
                    current_frame=current_frame,
                )

                sequence_loss = (
                    sequence_loss
                    + step_loss
                )

            # Average loss across all timesteps.
            sequence_loss = (
                sequence_loss
                / actions.shape[1]
            )

            # ----------------------------------------------------
            # Backpropagation
            # ----------------------------------------------------

            if training:

                sequence_loss.backward()

                clip_gradient_norm(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            total_loss += float(
                sequence_loss.item()
            )

            batches += 1

    if batches == 0:
        raise RuntimeError(
            "DataLoader produced no batches."
        )

    return (
        total_loss
        / batches
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(
    history,
    result_folder
):

    result_folder = Path(
        result_folder
    )

    result_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    csv_path = (
        result_folder
        / "training_history.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "best",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    # ---------------------------------------------------------
    # JSON
    # ---------------------------------------------------------

    json_path = (
        result_folder
        / "training_history.json"
    )

    json_path.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Training curve
    # ---------------------------------------------------------

    epochs = [
        row["epoch"]
        for row in history
    ]

    train_losses = [
        row["train_loss"]
        for row in history
    ]

    val_losses = [
        row["val_loss"]
        for row in history
    ]

    fig = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        label="Train",
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation",
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Motion-weighted MSE"
    )

    plt.title(
        "Final64 Teacher-Forced Sequential Training History"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        result_folder
        / "training_curve.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(
        args.seed
    )

    device = choose_device(
        args.device
    )

    # ========================================================
    # Directories
    # ========================================================

    data_folder = Path(
        DATA_FOLDER
    )

    model_folder = Path(
        MODEL_FOLDER
    )

    result_folder = Path(
        RESULT_FOLDER
    )

    model_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Prevent accidental overwrite
    # ========================================================

    if (
        MODEL_PATH.exists()
        and not args.overwrite
    ):

        raise FileExistsError(
            f"{MODEL_PATH} already exists. "
            "Use --overwrite only for an "
            "intentional fresh retraining."
        )

    # ========================================================
    # DATASET
    #
    # SequenceRangeDataset keeps the episode-safe sequence
    # filtering introduced in Final64.
    # ========================================================

    frame_count = len(
        np.load(
            data_folder / "frames.npy",
            mmap_mode="r",
        )
    )

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

    # ========================================================
    # Deterministic training shuffle
    # ========================================================

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # MODEL
    #
    # FINAL ARCHITECTURE:
    #
    # Encoder:
    #   64x64 grayscale
    #       -> Conv32
    #       -> Conv64
    #       -> Conv128
    #       -> Linear
    #       -> latent 128
    #
    # Dynamics:
    #   latent 128
    #       + action embedding 32
    #       -> GRU hidden 128
    #
    # Decoder:
    #   hidden 128
    #       -> 512
    #       -> 2048
    #       -> 4096
    #       -> 64x64
    # ========================================================

    model = WorldModel(
        latent_size=LATENT_SIZE,
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
    ).to(device)

    # ========================================================
    # LOSS
    #
    # Keep the known baseline settings.
    # ========================================================

    criterion = (
        StableMotionWeightedMSELoss(
            motion_weight=MOTION_WEIGHT,
            motion_threshold=MOTION_THRESHOLD,
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = Adam(
        list(model.parameters()),
        args.learning_rate,
    )

    parameter_count = sum(
        int(p.data.numel())
        for p in model.parameters()
    )

    # ========================================================
    # EXPERIMENT INFORMATION
    # ========================================================

    print("=" * 60)

    print(
        "FINAL64 TEACHER-FORCED SEQUENTIAL TRAINING"
    )

    print("=" * 60)

    print(
        "Total frames:",
        frame_count,
    )

    print(
        "Train frames:",
        TRAIN_END,
    )

    print(
        "Validation frames:",
        frame_count - TRAIN_END,
    )

    print(
        "Train valid sequences:",
        len(train_dataset),
    )

    print(
        "Validation valid sequences:",
        len(val_dataset),
    )

    print(
        "Model parameters:",
        f"{parameter_count:,}",
    )

    print(
        "Accelerator:",
        (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "CPU"
        ),
    )

    print(
        "Torch device:",
        device,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Sequence length:",
        SEQUENCE_LENGTH,
    )

    print(
        "Learning rate:",
        args.learning_rate,
    )

    print(
        "Motion weight:",
        MOTION_WEIGHT,
    )

    print(
        "Motion threshold:",
        MOTION_THRESHOLD,
    )

    print(
        "Maximum epochs:",
        args.epochs,
    )

    print(
        "Training mode:",
        "TEACHER-FORCED SEQUENTIAL",
    )

    print(
        "Episode-safe sequences:",
        "ENABLED",
    )

    print(
        "Best checkpoint criterion:",
        "VALIDATION LOSS",
    )

    print(
        "Early stopping:",
        "DISABLED",
    )

    print("=" * 60)

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_val = float("inf")
    best_epoch = 0

    history = []

    started = time.perf_counter()

    # ========================================================
    # TRAINING LOOP
    #
    # There is no early stopping.
    #
    # However, only the model with the lowest validation loss
    # is retained as best_world_model.npz.
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            training=False,
        )

        # ----------------------------------------------------
        # Best validation checkpoint
        # ----------------------------------------------------

        improved = (
            val_loss < best_val
        )

        if improved:

            best_val = val_loss
            best_epoch = epoch

            model.save(
                str(MODEL_PATH),
                epoch=epoch,
                best_loss=best_val,
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best": bool(improved),
            }
        )

        save_history(
            history,
            result_folder,
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f}"
        )

        if improved:

            print(
                "  Best checkpoint updated."
            )

        else:

            print(
                "  Validation did not improve. "
                "Continuing training."
            )

    # ========================================================
    # FINISHED
    # ========================================================

    elapsed = (
        time.perf_counter()
        - started
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "model": {

            "checkpoint":
                str(MODEL_PATH),

            "parameter_count":
                parameter_count,

            "latent_size":
                LATENT_SIZE,

            "hidden_size":
                HIDDEN_SIZE,

            "image_size":
                IMAGE_SIZE,
        },

        "training": {

            "seed":
                args.seed,

            "train_end":
                TRAIN_END,

            "validation_start":
                TRAIN_END,

            "frame_count":
                frame_count,

            "train_valid_sequences":
                len(train_dataset),

            "validation_valid_sequences":
                len(val_dataset),

            "sequence_length":
                SEQUENCE_LENGTH,

            "batch_size":
                args.batch_size,

            "learning_rate":
                args.learning_rate,

            "motion_weight":
                MOTION_WEIGHT,

            "motion_threshold":
                MOTION_THRESHOLD,

            "max_epochs":
                args.epochs,

            "early_stopping":
                False,

            "training_mode":
                "teacher_forced_sequential",

            "episode_safe_sequences":
                True,

            "checkpoint_selection":
                "lowest_validation_loss",

            "best_epoch":
                best_epoch,

            "best_validation_loss":
                best_val,

            "elapsed_seconds":
                elapsed,
        },

        "environment":
            environment_info(
                Path(__file__)
                .resolve()
                .parents[1]
            ),
    }

    write_json(
        result_folder
        / "training_metadata.json",
        metadata,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)

    print(
        "Epochs completed:",
        args.epochs,
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        best_val,
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "History:",
        result_folder
        / "training_history.csv",
    )

    print(
        "Elapsed seconds:",
        round(elapsed, 2),
    )

    print("=" * 60)


if __name__ == "__main__":
    main()