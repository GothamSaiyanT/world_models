import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from config_final64 import (
    ADAPTIVE_MOTION_THRESHOLD,
    ADAPTIVE_THRESHOLD,
    DATA_FOLDER,
    EVALUATION_HORIZON,
    EVALUATION_STARTS,
    FINAL_FRAMES_CSV,
    FINAL_METADATA_JSON,
    FINAL_REPORT_MD,
    FINAL_SUMMARY_CSV,
    FINAL_WINDOWS_CSV,
    FIXED_INTERVAL,
    HIDDEN_SIZE,
    IMAGE_SIZE,
    LATENT_SIZE,
    MODEL_PATH,
    RESULT_FOLDER,
)
from core.world_model import WorldModel
from training.final64_dataset import load_actions, load_normalized_frames
from utils.experiment_logging import environment_info, write_json
from utils.rollout_final64 import generate_rollout


PIPELINES = ("baseline", "fixed", "adaptive", "adaptive_motion")


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def evaluate_window(start, pipeline, result):
    real = result["real"]
    pred = result["prediction"]

    frame_rows = []
    for frame_index in range(1, len(real)):
        target = real[frame_index]
        prediction = pred[frame_index]
        difference = target - prediction

        mse = float(np.mean(difference ** 2))
        mae = float(np.mean(np.abs(difference)))
        psnr = psnr_from_mse(mse)
        ssim = float(
            structural_similarity(
                target,
                prediction,
                data_range=1.0,
            )
        )

        frame_rows.append(
            {
                "start": start,
                "pipeline": pipeline,
                "frame": frame_index,
                "dataset_index": start + frame_index,
                "mse": mse,
                "mae": mae,
                "psnr": psnr,
                "ssim": ssim,
                "correction_at_frame": frame_index in result["corrections"],
            }
        )

    frame_df = pd.DataFrame(frame_rows)

    real_motion_frames = np.mean(
        np.abs(real[1:] - real[:-1]),
        axis=(1, 2),
    )
    pred_motion_frames = np.mean(
        np.abs(pred[1:] - pred[:-1]),
        axis=(1, 2),
    )

    mean_real_motion = float(real_motion_frames.mean())
    mean_pred_motion = float(pred_motion_frames.mean())
    motion_ratio = float(mean_pred_motion / max(mean_real_motion, 1e-8))

    vacated_mask = (real[:-1] - real[1:]) > 0.03
    if vacated_mask.any():
        ghost_values = pred[1:][vacated_mask]
        ghost_brightness = float(ghost_values.mean())
        ghost_fraction = float((ghost_values > 0.10).mean())
    else:
        ghost_brightness = 0.0
        ghost_fraction = 0.0

    corrections = len(result["corrections"])
    horizon = len(real) - 1

    window_row = {
        "start": start,
        "pipeline": pipeline,
        "mse": float(frame_df["mse"].mean()),
        "mae": float(frame_df["mae"].mean()),
        "psnr": float(frame_df["psnr"].replace([np.inf, -np.inf], np.nan).mean()),
        "ssim": float(frame_df["ssim"].mean()),
        "real_motion": mean_real_motion,
        "predicted_motion": mean_pred_motion,
        "motion_ratio": motion_ratio,
        "ghost_brightness": ghost_brightness,
        "ghost_fraction_gt_010": ghost_fraction,
        "corrections": corrections,
        "correction_rate": float(corrections / max(horizon, 1)),
        "runtime_seconds": result["runtime_seconds"],
        "correction_frames": ",".join(str(x) for x in result["corrections"]),
    }

    return window_row, frame_rows


def make_report(
    summary_df,
    checkpoint_epoch,
    checkpoint_loss,
    starts,
    horizon,
    interval,
    threshold,
    motion_threshold,
):
    by_pipeline = summary_df.set_index("pipeline")
    baseline = by_pipeline.loc["baseline"]
    fixed = by_pipeline.loc["fixed"]
    adaptive = by_pipeline.loc["adaptive"]
    adaptive_motion = by_pipeline.loc["adaptive_motion"]

    fixed_mse_reduction = 100.0 * (baseline.mean_mse - fixed.mean_mse) / baseline.mean_mse
    adaptive_mse_reduction = 100.0 * (baseline.mean_mse - adaptive.mean_mse) / baseline.mean_mse
    adaptive_motion_mse_reduction = (
        100.0 * (baseline.mean_mse - adaptive_motion.mean_mse) / baseline.mean_mse
    )

    return f"""# Final64 experiment summary

## Frozen protocol

- Shared checkpoint epoch: {checkpoint_epoch}
- Checkpoint validation loss: {checkpoint_loss:.9f}
- Image resolution: 64x64 grayscale
- Evaluation starts: {list(starts)}
- Horizon: {horizon} predicted frames per window
- Fixed Interval correction: every {interval} frames, excluding a correction after the final prediction because it cannot affect a future step
- Adaptive correction threshold: whole-frame MSE > {threshold}
- Adaptive (motion) correction threshold: motion-weighted error > {motion_threshold}
- All four rollout strategies use the same learned world model.

## Five-window mean results

| Pipeline | Mean MSE | Mean MAE | Mean PSNR | Mean SSIM | Mean motion ratio | Mean effective corrections |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | {baseline.mean_mse:.6f} | {baseline.mean_mae:.6f} | {baseline.mean_psnr:.3f} | {baseline.mean_ssim:.4f} | {baseline.mean_motion_ratio:.3f}x | {baseline.mean_corrections:.2f} |
| Fixed Interval | {fixed.mean_mse:.6f} | {fixed.mean_mae:.6f} | {fixed.mean_psnr:.3f} | {fixed.mean_ssim:.4f} | {fixed.mean_motion_ratio:.3f}x | {fixed.mean_corrections:.2f} |
| Adaptive | {adaptive.mean_mse:.6f} | {adaptive.mean_mae:.6f} | {adaptive.mean_psnr:.3f} | {adaptive.mean_ssim:.4f} | {adaptive.mean_motion_ratio:.3f}x | {adaptive.mean_corrections:.2f} |
| Adaptive (motion) | {adaptive_motion.mean_mse:.6f} | {adaptive_motion.mean_mae:.6f} | {adaptive_motion.mean_psnr:.3f} | {adaptive_motion.mean_ssim:.4f} | {adaptive_motion.mean_motion_ratio:.3f}x | {adaptive_motion.mean_corrections:.2f} |

Relative to the Baseline: Fixed Interval reduced mean MSE by {fixed_mse_reduction:.1f}%, Adaptive by {adaptive_mse_reduction:.1f}%, and Adaptive (motion) by {adaptive_motion_mse_reduction:.1f}%, over these five frozen test windows.

## Interpretation to use in the paper

The experiment isolates rollout correction policy because the learned predictor, dataset, action sequence and evaluation windows are held constant. Baseline runs open-loop. Fixed Interval periodically re-anchors the next model input with an available observation. Adaptive re-anchors only when current prediction whole-frame MSE exceeds the frozen threshold. Adaptive (motion) re-anchors only when a motion-weighted error (the same weighting used during training, which emphasises pixels where the real scene changed) exceeds its own threshold. A correction does not replace the prediction that triggered it; that prediction remains in the scored rollout.

## Limitation

A correction strategy can reduce accumulated rollout error without correcting every one-step visual artefact. Motion ratio values above 1.0 indicate that the learned dynamics still exaggerate frame-to-frame change. Whole-frame-MSE-based adaptive correction assumes an observation is available so that current prediction error can be measured, but because the ball and paddle occupy a small fraction of the frame, a whole-frame error can stay below threshold even when they are completely mispredicted; this is the motivation for the motion-weighted variant.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--horizon", type=int, default=EVALUATION_HORIZON)
    parser.add_argument("--fixed-interval", type=int, default=FIXED_INTERVAL)
    parser.add_argument("--adaptive-threshold", type=float, default=ADAPTIVE_THRESHOLD)
    parser.add_argument(
        "--adaptive-motion-threshold",
        type=float,
        default=ADAPTIVE_MOTION_THRESHOLD,
    )
    parser.add_argument(
        "--starts",
        type=int,
        nargs="+",
        default=list(EVALUATION_STARTS),
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    frames = load_normalized_frames(DATA_FOLDER)
    actions = load_actions(DATA_FOLDER)

    model = WorldModel(
        latent_size=LATENT_SIZE,
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
    )
    checkpoint_epoch, checkpoint_loss = model.load(str(MODEL_PATH))
    model.to(device)
    model.eval()

    print("Device:", device)
    print("Frames:", frames.shape)
    print("Range:", frames.min(), frames.max())
    print("Loaded epoch:", checkpoint_epoch)
    print("Best validation loss:", checkpoint_loss)

    all_window_rows = []
    all_frame_rows = []

    for start in args.starts:
        if start < 0 or start + args.horizon >= len(frames):
            raise ValueError(f"Start {start} with horizon {args.horizon} exceeds dataset.")

        real = frames[start:start + args.horizon + 1]
        action_sequence = actions[start:start + args.horizon]

        print("\n" + "=" * 70)
        print("START", start)
        print("=" * 70)

        for pipeline in PIPELINES:
            result = generate_rollout(
                model=model,
                real_frames=real,
                actions=action_sequence,
                strategy=pipeline,
                fixed_interval=args.fixed_interval,
                adaptive_threshold=args.adaptive_threshold,
                adaptive_motion_threshold=args.adaptive_motion_threshold,
                device=device,
            )

            window_row, frame_rows = evaluate_window(start, pipeline, result)
            all_window_rows.append(window_row)
            all_frame_rows.extend(frame_rows)

            print("\n" + pipeline.upper())
            print("MSE:", f"{window_row['mse']:.8f}")
            print("MAE:", f"{window_row['mae']:.8f}")
            print("PSNR:", f"{window_row['psnr']:.3f}")
            print("SSIM:", f"{window_row['ssim']:.4f}")
            print("Motion ratio:", f"{window_row['motion_ratio']:.3f}x")
            print("Ghost brightness:", f"{window_row['ghost_brightness']:.6f}")
            print("Ghost >0.10:", f"{100.0 * window_row['ghost_fraction_gt_010']:.2f}%")
            print("Effective corrections:", result["corrections"])

    windows_df = pd.DataFrame(all_window_rows)
    frames_df = pd.DataFrame(all_frame_rows)

    summary_df = (
        windows_df.groupby("pipeline", as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            mean_mae=("mae", "mean"),
            mean_psnr=("psnr", "mean"),
            mean_ssim=("ssim", "mean"),
            mean_motion_ratio=("motion_ratio", "mean"),
            std_motion_ratio=("motion_ratio", "std"),
            mean_ghost_brightness=("ghost_brightness", "mean"),
            mean_ghost_fraction=("ghost_fraction_gt_010", "mean"),
            mean_corrections=("corrections", "mean"),
            mean_correction_rate=("correction_rate", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
    )

    Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(FINAL_WINDOWS_CSV, index=False)
    frames_df.to_csv(FINAL_FRAMES_CSV, index=False)
    summary_df.to_csv(FINAL_SUMMARY_CSV, index=False)

    metadata = {
        "checkpoint": {
            "path": str(MODEL_PATH),
            "epoch": int(checkpoint_epoch),
            "best_validation_loss": float(checkpoint_loss),
        },
        "protocol": {
            "starts": list(args.starts),
            "horizon": args.horizon,
            "fixed_interval": args.fixed_interval,
            "adaptive_threshold": args.adaptive_threshold,
            "adaptive_motion_threshold": args.adaptive_motion_threshold,
            "effective_correction_counting": "final-frame re-anchor is not counted because no future prediction follows",
            "seed_frame_scored": False,
        },
        "environment": environment_info(Path(__file__).resolve().parents[1]),
    }
    write_json(FINAL_METADATA_JSON, metadata)

    report = make_report(
        summary_df,
        checkpoint_epoch,
        checkpoint_loss,
        args.starts,
        args.horizon,
        args.fixed_interval,
        args.adaptive_threshold,
        args.adaptive_motion_threshold,
    )
    Path(FINAL_REPORT_MD).write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(FINAL_WINDOWS_CSV)
    print(FINAL_FRAMES_CSV)
    print(FINAL_SUMMARY_CSV)
    print(FINAL_METADATA_JSON)
    print(FINAL_REPORT_MD)


if __name__ == "__main__":
    main()
