from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise SystemExit(
        "scikit-image is required for SSIM. Install it with: pip install scikit-image"
    ) from exc

from training.dataset import WorldModelDataset
from utils.rollout_v3 import RolloutGeneratorV3
from v3.config import V3Config
from v3.device import resolve_accelerator
from v3.world_model import WorldModelV3


PIPELINES = ("baseline", "fixed_interval", "adaptive")


def load_pipeline(pipeline: str, model_folder: str, device):
    model_path = os.path.join(model_folder, f"best_{pipeline}_model.pt")
    checkpoint_path = os.path.join(
        model_folder, f"best_{pipeline}_checkpoint.pt"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = WorldModelV3(V3Config())
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    metadata = {}
    if os.path.exists(checkpoint_path):
        metadata = torch.load(checkpoint_path, map_location="cpu")

    return model, metadata


def to_frame_array(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().float().numpy()

    # V3 grayscale frames are normally [1, H, W].
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    return np.asarray(array, dtype=np.float32)


def data_range_for_pair(real: np.ndarray, pred: np.ndarray) -> float:
    maximum = max(float(real.max()), float(pred.max()))
    minimum = min(float(real.min()), float(pred.min()))

    # V3 frames normally live in [0, 1].
    if minimum >= -1e-6 and maximum <= 1.5:
        return 1.0

    # Fallback for image data stored like uint8.
    if minimum >= 0.0 and maximum <= 255.5:
        return 255.0

    return max(maximum - minimum, 1e-8)


def frame_metrics(real: torch.Tensor, pred: torch.Tensor):
    real_np = to_frame_array(real)
    pred_np = to_frame_array(pred)

    diff = pred_np - real_np
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))

    data_range = data_range_for_pair(real_np, pred_np)

    if mse <= 1e-15:
        psnr = float("inf")
    else:
        psnr = float(10.0 * math.log10((data_range * data_range) / mse))

    ssim = float(
        structural_similarity(
            real_np,
            pred_np,
            data_range=data_range,
        )
    )

    return mse, mae, psnr, ssim


@torch.no_grad()
def action_conditioned_difference(model, initial, device) -> float:
    outputs = []

    for action in range(4):
        hidden = model.init_hidden(1, device)
        output, _ = model(
            initial,
            torch.tensor([action], device=device, dtype=torch.long),
            hidden,
        )
        outputs.append(output.detach().cpu())

    differences = []

    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            differences.append(
                float(torch.abs(outputs[i] - outputs[j]).mean())
            )

    return float(np.mean(differences))


def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]

    if len(finite) == 0:
        return float("inf")

    return float(finite.mean())


def main():
    parser = argparse.ArgumentParser(
        description="Multi-start quantitative evaluation for V3 world models."
    )
    parser.add_argument(
        "--starts",
        type=int,
        nargs="+",
        default=[0, 500, 1000, 2000, 3000],
        help="Evaluation start indices.",
    )
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--data-folder", default="data_v3")
    parser.add_argument("--model-folder", default="models_v3")
    parser.add_argument("--output-folder", default="results_v3")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "tpu", "cpu"],
    )
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1.")

    runtime = resolve_accelerator(args.device)
    device = runtime.device

    print("Accelerator:", runtime.label)
    print("Starts:", args.starts)
    print("Horizon:", args.horizon)

    dataset = WorldModelDataset(args.data_folder)

    for start in args.starts:
        if start < 0:
            raise ValueError(f"Start cannot be negative: {start}")
        if start + args.horizon >= len(dataset):
            raise ValueError(
                f"start={start}, horizon={args.horizon} exceeds "
                f"dataset length {len(dataset)}."
            )

    loaded = {}

    for pipeline in PIPELINES:
        loaded[pipeline] = load_pipeline(
            pipeline, args.model_folder, device
        )

    window_rows = []
    frame_rows = []

    for start in args.starts:
        print(f"\n===== START {start} =====")

        initial = dataset[start][0].unsqueeze(0).to(device)

        actions = torch.stack(
            [
                dataset[start + index][1]
                for index in range(args.horizon)
            ]
        ).to(device).long()

        real = torch.stack(
            [
                dataset[start + index][0]
                for index in range(args.horizon + 1)
            ]
        )

        real_motion = torch.abs(
            real[1:] - real[:-1]
        ).mean(dim=(1, 2, 3)).numpy()

        for pipeline in PIPELINES:
            model, metadata = loaded[pipeline]

            threshold = metadata.get("adaptive_threshold")
            if threshold is None:
                threshold = metadata.get("validation_p90_drift")

            fixed_interval = int(metadata.get("fixed_interval", 8))

            generator = RolloutGeneratorV3(
                model,
                strategy=pipeline,
                fixed_interval=fixed_interval,
                adaptive_threshold=threshold,
                runtime=runtime,
            )

            runtime.sync()
            started = time.perf_counter()

            prediction = generator.generate(
                initial,
                actions,
                real_sequence=real if pipeline != "baseline" else None,
            )

            runtime.sync()
            rollout_seconds = time.perf_counter() - started
            prediction = prediction.detach().cpu()

            pred_motion = torch.abs(
                prediction[1:] - prediction[:-1]
            ).mean(dim=(1, 2, 3)).numpy()

            action_difference = action_conditioned_difference(
                model, initial, device
            )
            runtime.sync()

            mses = []
            maes = []
            psnrs = []
            ssims = []

            correction_set = set(generator.correction_steps)

            # Step 0 is the supplied seed frame, not a prediction.
            # Score only predicted future frames 1..horizon.
            for step in range(1, args.horizon + 1):
                mse, mae, psnr, ssim = frame_metrics(
                    real[step], prediction[step]
                )

                mses.append(mse)
                maes.append(mae)
                psnrs.append(psnr)
                ssims.append(ssim)

                frame_rows.append(
                    {
                        "start": start,
                        "pipeline": pipeline,
                        "step": step,
                        "dataset_frame": start + step,
                        "action": int(actions[step - 1].detach().cpu().item()),
                        "mse": mse,
                        "mae": mae,
                        "psnr": psnr,
                        "ssim": ssim,
                        "real_motion": float(real_motion[step - 1]),
                        "predicted_motion": float(pred_motion[step - 1]),
                        "corrected": int(step in correction_set),
                    }
                )

            mean_real_motion = float(real_motion.mean())
            mean_pred_motion = float(pred_motion.mean())
            motion_ratio = mean_pred_motion / max(mean_real_motion, 1e-12)
            correction_count = len(generator.correction_steps)
            correction_rate = correction_count / float(args.horizon)

            row = {
                "start": start,
                "pipeline": pipeline,
                "horizon": args.horizon,
                "mse": float(np.mean(mses)),
                "mae": float(np.mean(maes)),
                "psnr": finite_mean(psnrs),
                "ssim": float(np.mean(ssims)),
                "real_motion": mean_real_motion,
                "predicted_motion": mean_pred_motion,
                "motion_ratio": float(motion_ratio),
                "action_difference": action_difference,
                "corrections": correction_count,
                "correction_rate": correction_rate,
                "fixed_interval": fixed_interval
                if pipeline == "fixed_interval"
                else "",
                "adaptive_threshold": float(threshold)
                if threshold is not None
                else "",
                "rollout_seconds": float(rollout_seconds),
            }

            window_rows.append(row)

            print(
                f"{pipeline:14s} "
                f"MSE={row['mse']:.6f} "
                f"MAE={row['mae']:.6f} "
                f"PSNR={row['psnr']:.2f} "
                f"SSIM={row['ssim']:.4f} "
                f"motion={row['motion_ratio']:.2f}x "
                f"action_diff={row['action_difference']:.6f} "
                f"corrections={row['corrections']}"
            )

    summary_rows = []

    for pipeline in PIPELINES:
        selected = [
            row for row in window_rows
            if row["pipeline"] == pipeline
        ]

        def values(key):
            return np.asarray(
                [float(row[key]) for row in selected],
                dtype=np.float64,
            )

        summary_rows.append(
            {
                "pipeline": pipeline,
                "windows": len(selected),
                "mean_mse": float(values("mse").mean()),
                "std_mse": float(values("mse").std(ddof=0)),
                "mean_mae": float(values("mae").mean()),
                "std_mae": float(values("mae").std(ddof=0)),
                "mean_psnr": float(values("psnr").mean()),
                "std_psnr": float(values("psnr").std(ddof=0)),
                "mean_ssim": float(values("ssim").mean()),
                "std_ssim": float(values("ssim").std(ddof=0)),
                "mean_motion_ratio": float(values("motion_ratio").mean()),
                "std_motion_ratio": float(values("motion_ratio").std(ddof=0)),
                "mean_action_difference": float(
                    values("action_difference").mean()
                ),
                "std_action_difference": float(
                    values("action_difference").std(ddof=0)
                ),
                "mean_corrections": float(values("corrections").mean()),
                "std_corrections": float(values("corrections").std(ddof=0)),
                "mean_correction_rate": float(
                    values("correction_rate").mean()
                ),
                "mean_rollout_seconds": float(
                    values("rollout_seconds").mean()
                ),
            }
        )

    os.makedirs(args.output_folder, exist_ok=True)

    windows_path = os.path.join(
        args.output_folder, "v3_multistart_windows.csv"
    )
    frames_path = os.path.join(
        args.output_folder, "v3_multistart_frames.csv"
    )
    summary_path = os.path.join(
        args.output_folder, "v3_multistart_summary.csv"
    )

    write_csv(windows_path, window_rows)
    write_csv(frames_path, frame_rows)
    write_csv(summary_path, summary_rows)

    print("\n===== OVERALL SUMMARY =====")
    for row in summary_rows:
        print(
            f"{row['pipeline']:14s} "
            f"MSE={row['mean_mse']:.6f} "
            f"MAE={row['mean_mae']:.6f} "
            f"PSNR={row['mean_psnr']:.2f} "
            f"SSIM={row['mean_ssim']:.4f} "
            f"motion={row['mean_motion_ratio']:.2f}x "
            f"action_diff={row['mean_action_difference']:.6f} "
            f"corrections={row['mean_corrections']:.1f}"
        )

    print("\nSaved:")
    print(" ", windows_path)
    print(" ", frames_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
