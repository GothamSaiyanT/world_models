# 128x128 High-Resolution Retraining Path

This project keeps the original 64x64 experiment intact and adds a parallel
128x128 path for better rollout presentation and comparison.

## What changed

- `core/world_model.py`: baseline encoder now receives `image_size`, so 128x128
  inputs and outputs are dimensionally consistent.
- `pipelines/base.py`: passes the zero-based epoch into the self-correcting
  trainer so its existing 3-epoch warm-up logic actually runs.
- `.gitignore`: ignores generated high-resolution datasets, checkpoints,
  histories and videos.
- Added `highres_128.py`.
- Added `scripts/collect_128.py`.
- Added `scripts/smoke_test_128.py`.
- Added `scripts/train_128.py`.
- Added `scripts/train_adaptive_128.py`.
- Added `scripts/train_fixed_interval_128.py`.
- Added `scripts/render_128.py`.
- Added `comparison_dashboard_128.py`.
- Added `requirements_kaggle.txt`.

## Important

Do not resize the old 64x64 `frames.npy` and call that high resolution. Collect
fresh observations from the Atari source at 128x128. The new scripts write to
`data_128/`, `models_128/`, `results_128/`, and `outputs_128/`, leaving the
working 64x64 experiment untouched.

## Local architecture smoke test

```bash
python -m scripts.smoke_test_128
```

To also test the Atari environment and preprocessing:

```bash
python -m scripts.smoke_test_128 --collect-frames 50
```

## Collect the real high-resolution dataset

```bash
python -m scripts.collect_128 --steps 5000 --force
```

Verify that the result is `(5000, 128, 128)` for frames.

## Short training smoke test

Use this only to prove the whole path works; do not use it as final results.

```bash
python -m scripts.train_128 --epochs 1 --batch-size 2 --no-collect-if-missing
python -m scripts.train_adaptive_128 --epochs 1 --batch-size 2
python -m scripts.train_fixed_interval_128 --epochs 1 --batch-size 2
```

## Full training

Start conservatively with batch size 4 at 128x128:

```bash
python -m scripts.train_128 --epochs 70 --batch-size 4 --no-collect-if-missing
python -m scripts.train_adaptive_128 --epochs 70 --batch-size 4
python -m scripts.train_fixed_interval_128 --epochs 70 --batch-size 4
```

If GPU memory is comfortable, batch size 8 can be tested. Keep the same dataset
for all three pipelines.

## Render

```bash
python -m scripts.render_128 --pipeline baseline --start 0 --horizon 30
python -m scripts.render_128 --pipeline adaptive --start 0 --horizon 30
python -m scripts.render_128 --pipeline fixed_interval --start 0 --horizon 30
```

## Dashboard

After downloading `data_128/` and `models_128/` to your local PC:

```bash
python -m streamlit run comparison_dashboard_128.py
```
