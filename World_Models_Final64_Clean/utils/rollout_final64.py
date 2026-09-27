import time
import numpy as np
import torch

from core.loss import motion_weighted_error


@torch.no_grad()
def generate_rollout(
    model,
    real_frames,
    actions,
    strategy="baseline",
    fixed_interval=10,
    adaptive_threshold=0.0005,
    adaptive_motion_threshold=0.001,
    motion_weight=2.0,
    motion_threshold=0.05,
    device=None,
):
    """Generate a rollout using one shared learned world model.

    Strategies:
      baseline: fully open-loop after the seed frame.
      fixed: periodically re-anchor the *next input* using a real observation.
      adaptive: re-anchor the *next input* when current prediction whole-frame
                MSE exceeds a threshold and a real observation is available.
      adaptive_motion: like "adaptive", but the trigger is a motion-weighted
                error (same weighting used in training) instead of plain
                whole-frame MSE. Whole-frame MSE is dominated by the mostly
                static background, so it barely reacts when a small moving
                object (the ball, the paddle) is predicted in the wrong
                place. Weighting error by where the real scene actually
                moved makes the trigger sensitive to exactly those pixels.

    The prediction that triggers a correction is retained and scored. Corrections
    at the final horizon frame are intentionally not counted because they cannot
    affect any future prediction.

    Regardless of strategy, a per-step "diagnostics" list is returned with both
    the whole-frame MSE and the motion-weighted error at every step where a
    real next frame is available. This is meant for comparing strategies
    (e.g. plotting how each error signal behaves over the rollout, or checking
    how often "adaptive" should have corrected but didn't).
    """

    if strategy not in {"baseline", "fixed", "adaptive", "adaptive_motion"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_np = np.asarray(real_frames, dtype=np.float32)
    if real_np.max() > 1.0:
        real_np = real_np / 255.0

    action_np = np.asarray(actions, dtype=np.int64)

    horizon = len(action_np)
    if len(real_np) != horizon + 1:
        raise ValueError("real_frames must contain exactly horizon + 1 frames.")

    real = torch.from_numpy(real_np.copy()).float().unsqueeze(1).to(device)
    action_tensor = torch.from_numpy(action_np.copy()).long().to(device)

    hidden = model.init_hidden(batch_size=1, device=device)
    current = real[0].unsqueeze(0)

    predictions = [current.squeeze(0)]
    corrections = []
    correction_errors = {}
    diagnostics = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for step in range(horizon):
        prediction, hidden = model(
            current,
            action_tensor[step].view(1),
            hidden,
        )

        predictions.append(prediction.squeeze(0))
        next_input = prediction

        # A correction only has meaning if another prediction will follow.
        can_affect_future = (step + 1) < horizon

        # Always compute both error signals when a real next frame exists,
        # regardless of strategy, so the two can be compared afterwards.
        if can_affect_future:
            target = real[step + 1].unsqueeze(0)
            full_mse = torch.mean((prediction - target) ** 2).item()
            motion_error = motion_weighted_error(
                prediction,
                target,
                current,
                motion_weight=motion_weight,
                motion_threshold=motion_threshold,
            )
            diagnostics.append(
                {
                    "step": step + 1,
                    "full_mse": float(full_mse),
                    "motion_weighted_error": float(motion_error),
                }
            )

        if strategy == "fixed" and can_affect_future:
            if (step + 1) % int(fixed_interval) == 0:
                next_input = real[step + 1].unsqueeze(0)
                corrections.append(step + 1)

        elif strategy == "adaptive" and can_affect_future:
            if full_mse > float(adaptive_threshold):
                next_input = target
                corrections.append(step + 1)
                correction_errors[step + 1] = float(full_mse)

        elif strategy == "adaptive_motion" and can_affect_future:
            if motion_error > float(adaptive_motion_threshold):
                next_input = target
                corrections.append(step + 1)
                correction_errors[step + 1] = float(motion_error)

        current = next_input

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started

    predicted = torch.stack(predictions).detach().cpu().numpy()[:, 0]

    return {
        "prediction": predicted,
        "real": real_np,
        "corrections": corrections,
        "correction_errors": correction_errors,
        "diagnostics": diagnostics,
        "runtime_seconds": float(runtime),
    }
