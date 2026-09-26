# Final64 Breakout World Model

This is the cleaned final project used to train and evaluate a **64×64 Atari Breakout world model** and compare three rollout strategies with the **same learned predictor**:

- **Baseline** — fully open-loop after the seed frame.
- **Fixed Interval** — periodically re-anchors the next prediction input using an available observation.
- **Adaptive** — re-anchors the next prediction input only when current prediction MSE exceeds a frozen threshold.

The final design deliberately uses one shared model so the experiment isolates the correction policy rather than comparing three separately trained neural networks.

## Final frozen configuration

| Item | Setting |
|---|---|
| Environment | `ALE/Breakout-v5` |
| Dataset | 10,000 frames |
| Resolution | 64×64 grayscale |
| Seed | 42 |
| Action sampling | NOOP 10%, FIRE 10%, RIGHT 40%, LEFT 40% |
| Train/validation split | first 8,000 / final 2,000 frames |
| Sequence length | 16 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Motion-weighted MSE | weight 2.0, threshold 0.05 |
| Final evaluation starts | 0, 500, 1000, 2000, 3000 |
| Evaluation horizon | 100 |
| Fixed interval | 10 |
| Adaptive threshold | MSE > 0.0005 |

All defaults live in `config_final64.py` so the collector, trainer, evaluator, renderer and dashboard stay aligned.

## Kaggle quick start

Choose **GPU T4** in Kaggle, enable Internet, then run:

```bash
%cd /kaggle/working
!git clone -b final64-clean https://github.com/GothamSaiyanT/world_models.git
%cd /kaggle/working/world_models
!pip install -q -r requirements.txt
```

If you decide to make this clean project the `main` branch instead, omit `-b final64-clean`.

### 1. Smoke test

```bash
!python -m scripts.smoke_test_final64 --device cuda
```

Expected model size is approximately **10,375,008 parameters** and output shape `(2, 1, 64, 64)`.

### 2. Collect the final 10k dataset

```bash
!python -m scripts.collect_final64 --num-frames 10000 --seed 42
```

The dataset is stored compactly as `uint8`; training normalizes to `[0,1]` on load.

### 3. Validate the dataset

```bash
!python -m scripts.validate_dataset
```

This writes `results_final64/dataset_preview.png` and `dataset_validation.json`.

### 4. Train the shared predictor

```bash
!python -m scripts.train_final64 --device cuda --epochs 50 --patience 10 --batch-size 32
```

The best checkpoint is always:

```text
models_final64/best_world_model.npz
```

Do not select the final epoch manually; use the saved best-validation checkpoint.

### 5. Run the frozen five-window experiment

```bash
!python -m scripts.evaluate_final64 --device cuda
```

This generates:

```text
results_final64/final_multistart_windows.csv
results_final64/final_multistart_frames.csv
results_final64/final_multistart_summary.csv
results_final64/final_experiment_metadata.json
results_final64/final_research_summary.md
```

### 6. Render the presentation video

```bash
!python -m scripts.render_final64 --start 3000 --horizon 100 --fps 8 --device cuda
```

The 2×2 Ground Truth / Baseline / Fixed / Adaptive video is saved under `outputs_final64/`.

### 7. Export small paper-safe files

```bash
!python -m scripts.export_paper_results
```

The copied files go to `paper_results/`. Unlike data and checkpoints, this folder can be committed to Git after the final experiment if you want the numerical record version-controlled.

## Dashboard

On your local PC after downloading the trained checkpoint, dataset and result CSVs:

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard_final64.py
```

The dashboard shows synchronized Ground Truth/Baseline/Fixed/Adaptive rollouts, image metrics, correction timeline, temporal motion, error heatmaps and the frozen multi-start results.

## Important scientific interpretation

A correction changes the **input to the following prediction**. It does not replace the prediction that triggered the correction. The triggering prediction remains visible and scored.

Adaptive correction is **observation-error-triggered**: it needs an available real observation to measure prediction MSE. It is not a claim that the model can know its own true error without an observation.

See `research/EXPERIMENT_PROTOCOL.md` and `research/VIDEO_PRESENTATION_GUIDE.md` before writing the paper or recording the demo.
