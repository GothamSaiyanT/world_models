# V3 Motion-Fixed 128x128 World Models

V3 is an add-only experiment path created after the V2 diagnostic showed static-frame collapse and near-zero action sensitivity.

## What V3 changes

1. **Same PyTorch model for all three pipelines.** Only correction strategy changes.
2. **Convolutional decoder.** Avoids the enormous fully-connected 128x128 output layer.
3. **Motion-aware loss.** Moving pixels, temporal deltas, foreground pixels, and edges have separately normalised losses.
4. **32-step temporal training.** Longer than the previous 16-step training horizon.
5. **Scheduled autoregression.** Teacher forcing falls from 0.50 to 0.10 instead of feeding real frames at every step.
6. **Adaptive threshold calibration.** No hard-coded `0.05`; the threshold tracks the model's observed P90 drift error.
7. **Corrections are real during evaluation.** Fixed/Adaptive rollouts re-anchor the next state/input using the real observation. Baseline stays open-loop.
8. **Motion-richer dataset collection.** Breakout playfield is cropped and LEFT/RIGHT actions receive more sampling probability.

## Folders

V3 does not overwrite V2:

- `data_v3/`
- `models_v3/`
- `results_v3/`
- `outputs_v3/`

## Kaggle sequence

```bash
python -m scripts.smoke_test_v3
python -m scripts.collect_v3 --steps 10000
python -m scripts.train_v3 --pipeline baseline --epochs 60 --batch-size 4
python -m scripts.train_v3 --pipeline fixed_interval --epochs 60 --batch-size 4 --interval 8
python -m scripts.train_v3 --pipeline adaptive --epochs 60 --batch-size 4
python -m scripts.diagnose_v3 --horizon 100
```

Do not start a long run until the smoke test and a 2-epoch trial pass.

## Two-epoch trial

```bash
python -m scripts.train_v3 --pipeline baseline --epochs 2 --batch-size 2
python -m scripts.train_v3 --pipeline fixed_interval --epochs 2 --batch-size 2 --interval 8
python -m scripts.train_v3 --pipeline adaptive --epochs 2 --batch-size 2
```

Delete `models_v3/` and `results_v3/` after the trial before the real run.

## Rendering

```bash
python -m scripts.render_v3 --pipeline baseline --start 0 --horizon 100
python -m scripts.render_v3 --pipeline fixed_interval --start 0 --horizon 100
python -m scripts.render_v3 --pipeline adaptive --start 0 --horizon 100
```

Fixed/Adaptive are **observation-corrected** rollouts. Their correction mechanism uses ground-truth observations during evaluation by design; do not describe them as fully open-loop forecasts.


## TPU / GPU / CPU support

V3 training now accepts `--device auto|tpu|cuda|cpu`. `auto` prefers a normal CUDA GPU, otherwise checks for a TPU through PyTorch/XLA, and finally falls back to CPU. For Kaggle TPU commands see `V3_TPU_GPU_QUICKSTART.md`.
