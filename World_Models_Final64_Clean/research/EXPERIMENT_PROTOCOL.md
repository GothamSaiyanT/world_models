# Final Experiment Protocol

## Research question

How does rollout correction policy affect accumulated prediction error in a learned Atari Breakout world model when the underlying predictor is held constant?

## Controlled design

The final experiment uses **one shared learned 64×64 world model** for every rollout strategy. This removes model-training differences as a confounder.

### Controlled variables

- same world-model checkpoint
- same 10,000-frame dataset
- same action sequences
- same evaluation starts
- same 100-frame horizon
- same image preprocessing
- same metrics

### Independent variable

Rollout correction policy:

1. **Baseline** — open-loop autoregressive prediction after the seed frame.
2. **Fixed Interval** — periodically re-anchor the next model input using an available real observation.
3. **Adaptive** — re-anchor the next model input when current prediction MSE exceeds 0.0005.

### Dependent measures

- MSE
- MAE
- mean per-frame PSNR
- mean per-frame SSIM
- motion ratio = predicted frame-to-frame motion / real frame-to-frame motion
- departed-pixel ghost brightness
- proportion of departed pixels with predicted brightness > 0.10
- effective correction count/rate
- rollout runtime

## Frozen test windows

```text
0
500
1000
2000
3000
```

Each window predicts 100 future frames from one seed observation.

## Scoring rules

- Frame 0 is a shared real seed and is **not scored**.
- Predictions 1–100 are scored.
- A correction affects only the input to a future model step.
- The prediction that triggers a correction remains in the sequence and is scored.
- A re-anchor after the final prediction is not counted because it cannot affect any future prediction.

## Adaptive-policy limitation

Adaptive correction measures prediction error against an available real observation. It should be described as **observation-error-triggered correction**, not as self-knowledge of true error without observation.

## Avoid test-set tuning

The final values `fixed_interval=10` and `adaptive_threshold=0.0005` are frozen before the final five-window run. Do not repeatedly adjust them based on these five windows and then report the same windows as an untouched test set.

## Reproducibility record

The scripts automatically save:

- dataset metadata
- training metadata
- training history
- best checkpoint epoch and loss
- environment/package information
- Git commit hash where available
- window-level metrics
- frame-level metrics
- aggregate summary

Use these generated records when filling the research paper rather than copying numbers from memory.
