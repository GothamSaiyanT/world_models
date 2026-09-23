# Kaggle 128x128 Quick Start

Run these cells in order after pushing the updated code to GitHub.

## 1. Kaggle session settings

- Enable Internet for the setup/clone stage.
- Set Accelerator to GPU.

## 2. Clone the repository

```bash
!git clone https://github.com/GothamSaiyanT/world_models.git
%cd world_models
```

If the repository is already present in the session:

```bash
%cd /kaggle/working/world_models
!git pull origin main
```

## 3. Install the Atari dependency if needed

```bash
!pip install -q "gymnasium[atari]" ale-py
```

## 4. Confirm GPU

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Do not start a long training run unless `CUDA available` is `True`.

## 5. Architecture + Atari smoke test

```bash
!python -m scripts.smoke_test_128 --collect-frames 50
```

Expected end message:

```text
ALL 128x128 SMOKE TESTS PASSED
```

The collection test should also report frames shaped `(50, 128, 128)`.

## 6. Collect the real shared dataset

```bash
!python -m scripts.collect_128 --steps 5000 --force
```

For the final experiment, 10,000 frames can be used if training time and disk
budget allow. Do not recollect separately for each pipeline; all three must use
the same `data_128` files.

## 7. One-epoch end-to-end training test

```bash
!python -m scripts.train_128 --epochs 1 --batch-size 2 --no-collect-if-missing
!python -m scripts.train_adaptive_128 --epochs 1 --batch-size 2
!python -m scripts.train_fixed_interval_128 --epochs 1 --batch-size 2
```

Confirm these files exist:

```bash
!ls -lh models_128
!ls -lh results_128
```

## 8. Render a short test rollout

```bash
!python -m scripts.render_128 --pipeline baseline --start 0 --horizon 10
!python -m scripts.render_128 --pipeline adaptive --start 0 --horizon 10
!python -m scripts.render_128 --pipeline fixed_interval --start 0 --horizon 10
!ls -lh outputs_128
```

Only after all of the above passes should you start the real 70-epoch runs.
