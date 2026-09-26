# Kaggle Runbook

Use a fresh Kaggle notebook with **Accelerator: GPU T4** and Internet enabled.

## A. Clone and install

```python
%cd /kaggle/working
```

```bash
!git clone -b final64-clean https://github.com/GothamSaiyanT/world_models.git
```

```python
%cd /kaggle/working/world_models
```

```bash
!pip install -q -r requirements.txt
```

Verify GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## B. Run in this exact order

```bash
!python -m scripts.smoke_test_final64 --device cuda
!python -m scripts.collect_final64 --num-frames 10000 --seed 42
!python -m scripts.validate_dataset
!python -m scripts.train_final64 --device cuda --epochs 50 --patience 10 --batch-size 32
!python -m scripts.evaluate_final64 --device cuda
!python -m scripts.render_final64 --start 3000 --horizon 100 --fps 8 --device cuda
!python -m scripts.export_paper_results
```

## C. Checkpoint before closing Kaggle

The Kaggle `/working` directory is temporary. Before ending the session, create one archive containing the final generated experiment artifacts:

```bash
!zip -r final64_experiment_artifacts.zip \
    models_final64 \
    results_final64 \
    outputs_final64 \
    paper_results
```

Download `final64_experiment_artifacts.zip` from Kaggle.

The dataset can be regenerated from seed 42, so it does not need to be included unless you specifically want a local copy.

## D. Never push these to GitHub

Do not force-add:

```text
data_final64/
models_final64/
results_final64/
outputs_final64/
*.npy
*.npz
*.pt
*.mp4
```

The `.gitignore` already protects them.

## E. Files worth keeping for the paper

After `scripts.export_paper_results`, commit only the small files under `paper_results/` if desired. They record the final configuration, training curve and numerical results without pushing the large dataset or model.
