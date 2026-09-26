# Video Presentation Guide

A clear 4–6 minute recording is enough.

## 1. Problem — 30 to 45 seconds

Explain:

> A world model learns to predict what the environment will look like after an action. The difficult part is that small prediction errors accumulate when predicted frames are repeatedly fed back into the model. My project compares ways of controlling that accumulated rollout drift.

Show one Ground Truth vs Baseline sequence.

## 2. System architecture — 45 to 60 seconds

Explain the data flow:

```text
64×64 frame
   ↓
Encoder
   ↓
latent representation + action embedding
   ↓
GRU dynamics state
   ↓
Decoder
   ↓
predicted next frame
```

Emphasize that all three final pipelines use the same trained predictor.

## 3. Three rollout strategies — about 60 seconds

**Baseline**

> Each predicted frame becomes the input to the next prediction, so errors can accumulate.

**Fixed Interval**

> Every ten steps, an available real observation is used to re-anchor the next model input.

**Adaptive**

> The model is re-anchored only when prediction MSE against an available observation exceeds 0.0005.

Point to the dashboard correction timeline.

## 4. Final experiment — 60 to 90 seconds

Show the frozen five-window summary rather than selecting only a visually attractive sequence.

Explain:

- five starts: 0, 500, 1000, 2000, 3000
- 100 predicted frames per start
- same checkpoint and actions for all strategies
- frame 0 excluded from scoring

Discuss MSE, SSIM and correction counts first. Then mention motion ratio and ghosting as limitations.

## 5. Visual demo — 45 to 60 seconds

Play `outputs_final64/final64_start_3000_h100.mp4` or use the live dashboard.

Point out:

- Ground Truth
- Baseline drift
- periodic Fixed corrections
- irregular Adaptive corrections
- any remaining ball trail honestly

## 6. Conclusion — 30 seconds

Suggested structure:

> The experiment shows that observation-based re-anchoring can reduce accumulated world-model rollout error without retraining a different predictor for every policy. Adaptive correction changes when the system requests a real observation instead of correcting on a fixed schedule. The remaining limitation is that the learned one-step dynamics still exaggerate some motion, so correction reduces drift but does not remove every visual artefact.

Use the exact final CSV results when adding percentages.
