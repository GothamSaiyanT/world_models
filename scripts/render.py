"""Render baseline, fixed-interval, or adaptive world-model rollouts.

Examples
--------
Baseline:
    python -m scripts.render --pipeline baseline --start 8000 --horizon 200

Fixed interval:
    python -m scripts.render --pipeline fixed_interval --start 8000 --horizon 200

Adaptive:
    python -m scripts.render --pipeline adaptive --start 8000 --horizon 200
"""

import argparse
import json
import os

import torch

from config import ModelConfig
from training.dataset import WorldModelDataset
from utils.video import render_comparison_video


CHECKPOINTS = {
    "baseline": "models/best_world_model.npz",
    "fixed_interval": "models/best_fixed_interval_model.pt",
    "adaptive": "models/best_adaptive_model.pt",
}


def load_model(pipeline: str, checkpoint: str, device: torch.device):
    """Load the correct model implementation and checkpoint format."""

    if pipeline == "baseline":
        from core.world_model import WorldModel

        model = WorldModel(
            latent_size=128,
            hidden_size=128,
            image_size=64,
        )

        result = model.load(checkpoint)

        if isinstance(result, tuple):
            epoch, best_loss = result
            print(
                f"Baseline checkpoint epoch: {epoch}, "
                f"best loss: {best_loss}"
            )

    else:
        from core_nn.world_model import WorldModel

        config = ModelConfig(
            latent_size=128,
            hidden_size=128,
            image_size=64,
            input_channels=1,
            num_actions=4,
            action_embedding_size=32,
        )

        model = WorldModel(config)

        checkpoint_data = torch.load(
            checkpoint,
            map_location=device,
        )

        if (
            isinstance(checkpoint_data, dict)
            and "model_state_dict" in checkpoint_data
        ):
            state_dict = checkpoint_data["model_state_dict"]
        else:
            state_dict = checkpoint_data

        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def build_real_sequence_and_actions(
    dataset,
    start: int,
    horizon: int,
):
    """Collect the real comparison sequence and matching actions."""

    real_sequence = torch.stack([
        dataset[start + i][0]
        for i in range(horizon + 1)
    ])

    actions = torch.stack([
        torch.as_tensor(
            dataset[start + i][1]
        )
        for i in range(horizon)
    ]).long()

    return real_sequence, actions


def generate_baseline_rollout(
    model,
    initial_frame: torch.Tensor,
    actions: torch.Tensor,
):
    """Generate the original uncorrected baseline rollout."""

    from utils.rollout_generator import RolloutGenerator

    rollout = RolloutGenerator(model)

    return rollout.generate(
        initial_frame=initial_frame,
        actions=actions,
    )


def generate_corrected_rollout(
    model,
    real_sequence: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    pipeline: str,
    fixed_interval: int,
    adaptive_threshold: float,
):
    """
    Generate a rollout using the actual project correction classes.

    fixed_interval:
        Correct hidden state every fixed_interval steps.

    adaptive:
        Correct hidden state whenever per-sample MSE exceeds threshold.
    """

    from core_nn.correction import (
        AdaptiveCorrector,
        FixedIntervalCorrector,
    )
    from core_nn.drift import DriftDetector

    if pipeline == "fixed_interval":
        corrector = FixedIntervalCorrector(
            interval=fixed_interval
        )

    elif pipeline == "adaptive":
        corrector = AdaptiveCorrector(
            threshold=adaptive_threshold
        )

    else:
        raise ValueError(
            f"Corrected rollout is unsupported for pipeline: {pipeline}"
        )

    drift_detector = DriftDetector(
        metric="mse"
    )

    corrector.enabled = True
    corrector.reset_log()

    current_input = (
        real_sequence[0]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    hidden = model.init_hidden(
        batch_size=1,
        device=device,
    )

    predictions = [
        current_input.squeeze(0).cpu()
    ]

    drift_history = []

    with torch.no_grad():

        for step in range(len(actions)):

            action = (
                actions[step]
                .reshape(1)
                .to(
                    device=device,
                    dtype=torch.long,
                )
            )

            latent = model.encode(
                current_input
            )

            prediction, hidden = model.step(
                latent,
                action,
                hidden,
            )

            predictions.append(
                prediction
                .squeeze(0)
                .cpu()
            )

            real_next = (
                real_sequence[step + 1]
                .unsqueeze(0)
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            error = drift_detector.compute_error(
                prediction,
                real_next,
            )

            drift_history.append(
                float(error.mean().item())
            )

            hidden = corrector.maybe_correct(
                hidden=hidden,
                real_frame=real_next,
                encode_fn=model.encode,
                error=error,
                step=step,
            )

            # Autoregressive rollout:
            # next visual input remains the prediction.
            current_input = prediction.detach()

    predicted_sequence = torch.stack(
        predictions
    )

    return (
        predicted_sequence,
        corrector.correction_log,
        drift_history,
    )


def save_rollout_metadata(
    output_dir: str,
    pipeline: str,
    start: int,
    horizon: int,
    correction_log,
    drift_history,
    fixed_interval: int,
    adaptive_threshold: float,
):
    """Save correction/drift information beside the video."""

    metadata = {
        "pipeline": pipeline,
        "start": start,
        "horizon": horizon,
        "fixed_interval": fixed_interval,
        "adaptive_threshold": adaptive_threshold,
        "correction_events": [
            {
                "step_zero_based": int(step),
                "prediction_number": int(step) + 1,
                "mean_error": float(mean_error),
                "corrected_samples": int(corrected_samples),
            }
            for (
                step,
                mean_error,
                corrected_samples,
            ) in correction_log
        ],
        "drift_history": drift_history,
    }

    path = os.path.join(
        output_dir,
        f"rollout_metadata_{pipeline}.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "Saved rollout metadata to:",
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline",
        choices=[
            "baseline",
            "fixed_interval",
            "adaptive",
        ],
        default="baseline",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--fixed-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    if args.fixed_interval <= 0:
        raise ValueError(
            "fixed-interval must be greater than zero."
        )

    if args.adaptive_threshold < 0:
        raise ValueError(
            "adaptive-threshold cannot be negative."
        )

    checkpoint = CHECKPOINTS[
        args.pipeline
    ]

    if not os.path.exists(
        checkpoint
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Pipeline:",
        args.pipeline,
    )

    print(
        "Rendering on:",
        device,
    )

    print(
        "Checkpoint:",
        checkpoint,
    )

    dataset = WorldModelDataset(
        folder="data"
    )

    if args.start < 0:
        raise ValueError(
            "start cannot be negative."
        )

    if (
        args.start
        + args.horizon
        >= len(dataset)
    ):
        raise ValueError(
            f"Requested frames up to "
            f"{args.start + args.horizon}, "
            f"but dataset contains "
            f"{len(dataset)} samples."
        )

    real_sequence, actions = (
        build_real_sequence_and_actions(
            dataset,
            args.start,
            args.horizon,
        )
    )

    print(
        "Real sequence shape:",
        real_sequence.shape,
    )

    print(
        "Actions shape:",
        actions.shape,
    )

    model = load_model(
        args.pipeline,
        checkpoint,
        device,
    )

    if args.pipeline == "baseline":

        initial_frame = (
            real_sequence[0]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        predicted_sequence = (
            generate_baseline_rollout(
                model,
                initial_frame,
                actions.to(device),
            )
        )

        correction_log = []
        drift_history = []

    else:

        (
            predicted_sequence,
            correction_log,
            drift_history,
        ) = generate_corrected_rollout(
            model=model,
            real_sequence=real_sequence,
            actions=actions,
            device=device,
            pipeline=args.pipeline,
            fixed_interval=(
                args.fixed_interval
            ),
            adaptive_threshold=(
                args.adaptive_threshold
            ),
        )

    print(
        "Predicted sequence shape:",
        predicted_sequence.shape,
    )

    if args.pipeline != "baseline":

        print(
            "Number of correction events:",
            len(correction_log),
        )

        if correction_log:
            print(
                "First correction events:"
            )

            for event in correction_log[:10]:
                step, error, samples = event

                print(
                    f"  prediction={step + 1}, "
                    f"error={error:.6f}, "
                    f"samples={samples}"
                )

    os.makedirs(
        "outputs",
        exist_ok=True,
    )

    output_path = (
        f"outputs/"
        f"rollout_comparison_"
        f"{args.pipeline}.mp4"
    )

    render_comparison_video(
        real_sequence,
        predicted_sequence,
        output_path,
        fps=args.fps,
    )

    print(
        "\nSaved comparison video to:",
        output_path,
    )

    if args.pipeline != "baseline":
        save_rollout_metadata(
            output_dir="outputs",
            pipeline=args.pipeline,
            start=args.start,
            horizon=args.horizon,
            correction_log=correction_log,
            drift_history=drift_history,
            fixed_interval=(
                args.fixed_interval
            ),
            adaptive_threshold=(
                args.adaptive_threshold
            ),
        )


if __name__ == "__main__":
    main()
