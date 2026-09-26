# Paper Results Template

Do not fill this file before the final Kaggle evaluation. Use `results_final64/final_research_summary.md` and the generated CSV files as the numerical source.

## Experimental setup

A single learned 64×64 Atari Breakout world model was evaluated under three rollout policies: open-loop Baseline, periodic Fixed Interval observation correction, and Adaptive observation-error-triggered correction. All policies used the same checkpoint, actions, starting frames and 100-frame horizons.

## Quantitative results

Insert the table generated in:

```text
results_final64/final_research_summary.md
```

Report means across the five frozen starts and, where useful, the standard deviations from `final_multistart_summary.csv`.

## What to discuss

- whether correction reduced MSE/MAE compared with Baseline
- whether SSIM/PSNR changed consistently
- how many effective corrections Fixed and Adaptive used
- whether Adaptive used observations less frequently than Fixed
- whether motion ratio remained above 1.0
- whether ghost metrics improved or worsened
- sequence-to-sequence variability across starts

## Safe interpretation language

Use language such as:

> Over the five frozen 100-frame test windows, observation correction reduced accumulated rollout error relative to the open-loop Baseline.

If supported by the final numbers, add:

> Adaptive correction achieved comparable image-level accuracy while using fewer observation re-anchors on average than the periodic policy.

Do **not** state that correction removed all ghosting unless the final ghost metrics and video actually support that claim.

## Limitations

At minimum discuss:

- small Atari-specific dataset
- single game/environment
- grayscale 64×64 representation
- random/action-biased data collection rather than expert gameplay
- residual long-horizon motion artefacts
- adaptive error calculation requires access to a real observation
- five evaluation windows are useful but limited for broad generalisation
