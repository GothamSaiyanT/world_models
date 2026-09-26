# Project Map

## End-to-end flow

```text
ALE/Breakout-v5
      ↓
scripts/collect_final64.py
      ↓
data_final64/frames.npy + actions.npy
      ↓
training/final64_dataset.py
      ↓
core/world_model.py
      ↓
models_final64/best_world_model.npz
      ↓
utils/rollout_final64.py
      ↓
Baseline / Fixed Interval / Adaptive
      ↓
scripts/evaluate_final64.py
      ↓
results_final64/*.csv + report
      ↓
dashboard_final64.py / scripts/render_final64.py
```

## Neural-network files

### `core/encoder.py`
Input: one 64×64 grayscale frame.

Process: three custom convolution layers reduce the image to a compact feature representation, then a linear layer maps it to a 128-dimensional latent vector.

Output: latent representation of the current visual state.

### `core/dynamics.py`
Input: latent state, action, previous recurrent hidden state.

Process: the action is embedded into a vector, concatenated with the visual latent representation, and passed through a custom GRU cell.

Output: updated hidden state representing the predicted dynamics.

### `core/decoder.py`
Input: recurrent hidden state.

Process: fully connected layers expand the hidden state back to 4,096 pixel values. Sigmoid keeps predictions in the 0–1 range.

Output: predicted next 64×64 grayscale frame.

### `core/world_model.py`
Connects Encoder → Dynamics → Decoder. It also saves/loads the custom model checkpoint.

### `core/loss.py`
Motion-weighted MSE. Moving pixels receive extra weight so the model does not learn only the mostly static Breakout background.

## Data and training files

### `scripts/collect_final64.py`
Downloads/uses the Atari environment through ALE, collects 10,000 Breakout observations with the frozen action distribution, preprocesses them to 64×64 grayscale, and stores compact uint8 arrays.

### `training/final64_dataset.py`
Reads contiguous sequences for recurrent training. It automatically converts uint8 frames to float32 and divides by 255 so the model sees values in [0,1]. This is the normalization bug that was explicitly fixed during development.

### `scripts/train_final64.py`
Trains one shared world model using sequences of 16 transitions, an 8,000/2,000 train-validation split, validation-based checkpointing and early stopping.

## Rollout strategy files

### `utils/rollout_final64.py`
Contains the three inference policies around the same world model.

- Baseline feeds predictions back into the model continuously.
- Fixed Interval uses a real observation every 10 steps to re-anchor the next prediction input.
- Adaptive checks current prediction MSE against an available observation and re-anchors when MSE > 0.0005.

A triggering prediction is never overwritten. It remains visible and scored.

## Evaluation and presentation files

### `scripts/evaluate_final64.py`
Runs the frozen five-start experiment, calculates MSE, MAE, PSNR, SSIM, motion ratio, ghost metrics, correction counts and runtime, then writes CSV/JSON/Markdown research records.

### `dashboard_final64.py`
Interactive presentation interface showing synchronized rollouts, metrics, motion, correction timeline and error heatmaps.

### `scripts/render_final64.py`
Creates a 2×2 MP4 with Ground Truth, Baseline, Fixed Interval and Adaptive for presentation recording.

### `scripts/export_paper_results.py`
Copies only small experiment records to `paper_results/` so they can be version-controlled without committing model/data binaries.
