# V3 Accelerator Quickstart (Kaggle)

V3 now supports the same command on:

- TPU through PyTorch/XLA
- CUDA GPU through normal PyTorch
- CPU as a fallback

## 1. Confirm Kaggle accelerator

If the notebook is attached to a TPU, run:

```python
import torch
print("CUDA:", torch.cuda.is_available())

import torch_xla
import torch_xla.runtime as xr
print("XLA device type:", xr.device_type())
print("XLA devices:", torch_xla.real_devices())
```

For TPU you want `XLA device type: TPU`.

## 2. TPU smoke test

```bash
!python -m scripts.smoke_test_v3 --device tpu
```

Expected ending:

```text
Accelerator: TPU/XLA (...)
Prediction: (2, 1, 128, 128)
V3 SMOKE TEST PASSED
```

The first XLA execution compiles a graph, so the very first step can be much slower than later steps.

## 3. Short TPU training test

Start with a larger batch than the previous CPU test. TPU throughput is poor with tiny batches.

```bash
!python -m scripts.train_v3 \
  --pipeline baseline \
  --epochs 2 \
  --batch-size 8 \
  --device tpu
```

If memory is comfortable, try `--batch-size 16`. If it runs out of memory, use `4`.

Then train the other two test models:

```bash
!python -m scripts.train_v3 --pipeline fixed_interval --epochs 2 --batch-size 8 --interval 8 --device tpu
!python -m scripts.train_v3 --pipeline adaptive --epochs 2 --batch-size 8 --device tpu
```

## 4. Diagnose motion before long training

```bash
!python -m scripts.diagnose_v3 --horizon 50 --device tpu
```

Do not commit to a long run unless predicted motion continues beyond the first step and action-conditioned differences are non-zero.

## 5. Long training

Only after the 2-epoch test passes:

```bash
!python -m scripts.train_v3 --pipeline baseline --epochs 60 --batch-size 8 --device tpu
!python -m scripts.train_v3 --pipeline fixed_interval --epochs 60 --batch-size 8 --interval 8 --device tpu
!python -m scripts.train_v3 --pipeline adaptive --epochs 60 --batch-size 8 --device tpu
```

## GPU alternative

With a Kaggle GPU selected, either use `--device auto` or force CUDA:

```bash
!python -m scripts.train_v3 --pipeline baseline --epochs 2 --batch-size 8 --device cuda
```

`auto` prefers CUDA when CUDA exists, otherwise it checks for TPU/XLA, and then falls back to CPU.

## Why this version is faster on TPU than the previous V3 trainer

The old trainer repeatedly converted XLA tensors to Python numbers inside the 32-step recurrent loop. That forces synchronization between the host and TPU. The updated trainer keeps correction masks and metrics on the accelerator, uses `MpDeviceLoader`, uses XLA-aware optimizer steps, samples drift statistics instead of reading them every step, and saves XLA checkpoints in CPU-compatible form.
