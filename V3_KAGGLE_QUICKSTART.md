# V3 Kaggle Quickstart

## 1. Pull the updated repository

```bash
git pull origin main
```

or clone into a fresh Kaggle session:

```bash
git clone https://github.com/GothamSaiyanT/world_models.git
cd world_models
```

## 2. Install Atari dependencies

```bash
pip install -q "gymnasium[atari]" ale-py opencv-python-headless
```

## 3. Architecture smoke test

```bash
python -m scripts.smoke_test_v3
```

Expected final line:

```text
V3 SMOKE TEST PASSED
```

## 4. Collect a fresh motion-richer dataset

Do not reuse `data_128` for the final V3 run.

```bash
python -m scripts.collect_v3 --steps 10000
```

This creates:

```text
data_v3/frames.npy
data_v3/actions.npy
```

## 5. Two-epoch end-to-end trial

```bash
python -m scripts.train_v3 --pipeline baseline --epochs 2 --batch-size 2 && \
python -m scripts.train_v3 --pipeline fixed_interval --epochs 2 --batch-size 2 --interval 8 && \
python -m scripts.train_v3 --pipeline adaptive --epochs 2 --batch-size 2
```

Then diagnose:

```bash
python -m scripts.diagnose_v3 --horizon 50
```

The critical check is that predicted motion does **not** become exactly zero after the first step and action-conditioned differences are non-zero.

After the trial, remove its checkpoints/history before the long run:

```bash
rm -rf models_v3 results_v3 outputs_v3
```

Keep `data_v3`.

## 6. Long training

Start with batch size 4. If Kaggle GPU memory is comfortable, batch size 8 can be tested.

```bash
python -m scripts.train_v3 --pipeline baseline --epochs 60 --batch-size 4 && \
python -m scripts.train_v3 --pipeline fixed_interval --epochs 60 --batch-size 4 --interval 8 && \
python -m scripts.train_v3 --pipeline adaptive --epochs 60 --batch-size 4
```

All three use the same architecture, same train/validation split, same loss, and same dataset.

## 7. Diagnose before long rollouts

```bash
python -m scripts.diagnose_v3 --horizon 100
```

Do not proceed to project-day experiments if motion again collapses to zero.

## 8. Render 100 / 300 / 600

```bash
python -m scripts.render_v3 --pipeline baseline --start 0 --horizon 100 && \
python -m scripts.render_v3 --pipeline fixed_interval --start 0 --horizon 100 && \
python -m scripts.render_v3 --pipeline adaptive --start 0 --horizon 100 && \
python -m scripts.render_v3 --pipeline baseline --start 0 --horizon 300 && \
python -m scripts.render_v3 --pipeline fixed_interval --start 0 --horizon 300 && \
python -m scripts.render_v3 --pipeline adaptive --start 0 --horizon 300 && \
python -m scripts.render_v3 --pipeline baseline --start 0 --horizon 600 && \
python -m scripts.render_v3 --pipeline fixed_interval --start 0 --horizon 600 && \
python -m scripts.render_v3 --pipeline adaptive --start 0 --horizon 600
```

V3 render filenames include pipeline/start/horizon, so they do not overwrite each other.

## 9. Package outputs

```bash
zip -r world_models_v3_final.zip data_v3 models_v3 results_v3 outputs_v3
```

## 10. Local dashboard

Copy the four V3 folders into the project root, then:

```cmd
python -m streamlit run comparison_dashboard_v3.py
```

### Interpretation note

V3 Baseline is open-loop. V3 Fixed and Adaptive are observation-corrected pipelines; they use real observations at correction points. State this explicitly in the report/presentation.
