# V4 — Balanced Motion 64×64

V4 keeps the useful lessons from V3 but returns to 64×64 for faster, more stable experimentation.

## Why V4 exists

- Original 64×64: static/frozen prediction collapse.
- V3 128×128: motion and action sensitivity were restored, but motion was over-emphasized and the decoder left ghost trails behind the ball/paddle.
- V4 64×64: balanced motion supervision + a targeted ghost penalty + a smaller upsample/convolution decoder.

V4 uses new folders and does not overwrite V3:

- `data_v4_64/`
- `models_v4_64/`
- `results_v4_64/`
- `outputs_v4_64/`

## 1. Prepare 64×64 data from the existing V3 10k frames

```bash
python -m scripts.prepare_v4_64 --source data_v3 --output data_v4_64
```

## 2. Smoke test

GPU:

```bash
python -m scripts.smoke_test_v4 --device cuda
```

TPU:

```bash
python -m scripts.smoke_test_v4 --device tpu
```

CPU:

```bash
python -m scripts.smoke_test_v4 --device cpu
```

## 3. Do a short 3-epoch baseline test FIRST

```bash
python -m scripts.train_v4 --pipeline baseline --epochs 3 --batch-size 16 --device cuda
```

Then diagnose:

```bash
python -m scripts.diagnose_v4 --start 3000 --horizon 50 --device cuda
```

Render the short rollout:

```bash
python -m scripts.render_v4 --pipeline baseline --start 3000 --horizon 50 --device cuda
```

Inspect the paused frames. We specifically want one ball and one paddle, not a dotted trail.

## 4. Only after the short test looks healthy

Train all pipelines with the same maximum budget and early stopping:

```bash
python -m scripts.train_v4 --pipeline baseline --epochs 30 --batch-size 16 --patience 8 --device cuda && \
python -m scripts.train_v4 --pipeline fixed_interval --epochs 30 --batch-size 16 --interval 8 --patience 8 --device cuda && \
python -m scripts.train_v4 --pipeline adaptive --epochs 30 --batch-size 16 --patience 8 --device cuda
```

Do not start this full run until the 3-epoch baseline visual test is checked.
