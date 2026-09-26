import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from skimage.metrics import structural_similarity

from config_final64 import (
    ADAPTIVE_THRESHOLD,
    DATA_FOLDER,
    EVALUATION_HORIZON,
    FINAL_SUMMARY_CSV,
    FINAL_WINDOWS_CSV,
    FIXED_INTERVAL,
    HIDDEN_SIZE,
    IMAGE_SIZE,
    LATENT_SIZE,
    MODEL_PATH,
)
from core.world_model import WorldModel
from training.final64_dataset import load_actions, load_normalized_frames
from utils.rollout_final64 import generate_rollout


st.set_page_config(
    page_title="Final64 World Model Comparison",
    page_icon="🧠",
    layout="wide",
)

st.title("Final64 World Model — Rollout Strategy Comparison")
st.caption(
    "One shared 64×64 Breakout world model. Baseline, Fixed Interval and "
    "Adaptive differ only in how observations are used to correct the rollout."
)

PIPELINES = ("baseline", "fixed", "adaptive")
LABELS = {
    "baseline": "Baseline",
    "fixed": "Fixed Interval",
    "adaptive": "Adaptive",
}


@st.cache_resource(show_spinner="Loading dataset...")
def cached_dataset():
    return load_normalized_frames(DATA_FOLDER), load_actions(DATA_FOLDER)


@st.cache_resource(show_spinner="Loading trained world model...")
def cached_model(device_name):
    device = torch.device(device_name)
    model = WorldModel(
        latent_size=LATENT_SIZE,
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
    )
    epoch, best_loss = model.load(str(MODEL_PATH))
    model.to(device)
    model.eval()
    return model, int(epoch), float(best_loss)


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def per_frame_metrics(real, prediction, pipeline):
    rows = []
    for frame_index in range(1, min(len(real), len(prediction))):
        target = real[frame_index]
        pred = prediction[frame_index]
        diff = target - pred
        mse = float(np.mean(diff ** 2))
        rows.append(
            {
                "pipeline": pipeline,
                "frame": frame_index,
                "mse": mse,
                "mae": float(np.mean(np.abs(diff))),
                "psnr": psnr_from_mse(mse),
                "ssim": float(structural_similarity(target, pred, data_range=1.0)),
            }
        )
    return pd.DataFrame(rows)


def sequence_summary(real, prediction, corrections, runtime):
    frame_df = per_frame_metrics(real, prediction, "tmp")

    real_motion = float(
        np.mean(np.abs(real[1:] - real[:-1]), axis=(1, 2)).mean()
    )
    predicted_motion = float(
        np.mean(np.abs(prediction[1:] - prediction[:-1]), axis=(1, 2)).mean()
    )

    vacated = (real[:-1] - real[1:]) > 0.03
    if vacated.any():
        ghost_values = prediction[1:][vacated]
        ghost_brightness = float(ghost_values.mean())
        ghost_fraction = float((ghost_values > 0.10).mean())
    else:
        ghost_brightness = 0.0
        ghost_fraction = 0.0

    return {
        "MSE": float(frame_df["mse"].mean()),
        "MAE": float(frame_df["mae"].mean()),
        "Mean PSNR": float(
            frame_df["psnr"].replace([np.inf, -np.inf], np.nan).mean()
        ),
        "Mean SSIM": float(frame_df["ssim"].mean()),
        "Motion Ratio": predicted_motion / max(real_motion, 1e-8),
        "Ghost Brightness": ghost_brightness,
        "Ghost >0.10": ghost_fraction,
        "Effective Corrections": int(len(corrections)),
        "Correction Rate": len(corrections) / max(len(real) - 1, 1),
        "Runtime (s)": float(runtime),
    }


def metric_figure(metrics, metric, current_frame):
    fig = go.Figure()
    for pipeline in PIPELINES:
        subset = metrics[metrics["pipeline"] == pipeline]
        fig.add_trace(
            go.Scatter(
                x=subset["frame"],
                y=subset[metric],
                mode="lines",
                name=LABELS[pipeline],
            )
        )
    fig.add_vline(x=current_frame, line_dash="dot")
    fig.update_layout(
        title=f"{metric.upper()} over rollout",
        xaxis_title="Predicted frame",
        yaxis_title=metric.upper(),
        hovermode="x unified",
        height=420,
    )
    return fig


def correction_figure(results, horizon):
    fig = go.Figure()
    positions = {"baseline": 0, "fixed": 1, "adaptive": 2}

    for pipeline in PIPELINES:
        corrections = results[pipeline]["corrections"]
        if corrections:
            text = []
            for frame in corrections:
                if pipeline == "adaptive":
                    error = results[pipeline]["correction_errors"].get(frame)
                    text.append(f"Frame {frame}<br>MSE {error:.6f}" if error is not None else f"Frame {frame}")
                else:
                    text.append(f"Frame {frame}")

            fig.add_trace(
                go.Scatter(
                    x=corrections,
                    y=[positions[pipeline]] * len(corrections),
                    mode="markers",
                    marker=dict(size=13),
                    name=LABELS[pipeline],
                    text=text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Baseline", "Fixed Interval", "Adaptive"],
        range=[-0.5, 2.5],
    )
    fig.update_xaxes(title="Rollout frame", range=[0, horizon])
    fig.update_layout(title="Effective correction timeline", height=300, showlegend=False)
    return fig


def motion_figure(results, current_frame):
    fig = go.Figure()
    real = results["baseline"]["real"]
    real_motion = np.mean(np.abs(real[1:] - real[:-1]), axis=(1, 2))

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(real_motion) + 1),
            y=real_motion,
            mode="lines",
            name="Ground Truth",
        )
    )

    for pipeline in PIPELINES:
        pred = results[pipeline]["prediction"]
        motion = np.mean(np.abs(pred[1:] - pred[:-1]), axis=(1, 2))
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(motion) + 1),
                y=motion,
                mode="lines",
                name=LABELS[pipeline],
            )
        )

    fig.add_vline(x=current_frame, line_dash="dot")
    fig.update_layout(
        title="Frame-to-frame motion",
        xaxis_title="Frame",
        yaxis_title="Mean absolute frame change",
        hovermode="x unified",
        height=420,
    )
    return fig


if not Path(MODEL_PATH).exists():
    st.error(f"Missing trained checkpoint: `{MODEL_PATH}`")
    st.stop()

if not Path(DATA_FOLDER, "frames.npy").exists():
    st.error(f"Missing final dataset: `{DATA_FOLDER}/frames.npy`")
    st.stop()

frames, actions = cached_dataset()

device_options = ["cpu"]
if torch.cuda.is_available():
    device_options.insert(0, "cuda")

st.sidebar.header("Experiment controls")
device_name = st.sidebar.selectbox("Device", device_options, index=0)
horizon = st.sidebar.select_slider(
    "Rollout horizon",
    options=[25, 50, 100, 150, 200],
    value=EVALUATION_HORIZON,
)
max_start = max(0, len(frames) - horizon - 1)
start = st.sidebar.number_input(
    "Start frame",
    min_value=0,
    max_value=max_start,
    value=min(3000, max_start),
    step=1,
)
fixed_interval = st.sidebar.number_input(
    "Fixed interval",
    min_value=1,
    max_value=max(1, horizon),
    value=min(FIXED_INTERVAL, horizon),
    step=1,
)
adaptive_threshold = st.sidebar.number_input(
    "Adaptive MSE threshold",
    min_value=0.00001,
    max_value=0.01,
    value=ADAPTIVE_THRESHOLD,
    step=0.00005,
    format="%.5f",
)
playback_speed = st.sidebar.select_slider(
    "Playback speed",
    options=[0.25, 0.5, 1.0, 2.0, 4.0],
    value=0.5,
)
run_button = st.sidebar.button("Run comparison", type="primary", use_container_width=True)

model, epoch, best_loss = cached_model(device_name)

a, b, c, d = st.columns(4)
a.metric("Checkpoint", f"Epoch {epoch}")
b.metric("Validation loss", f"{best_loss:.6f}")
c.metric("Resolution", "64 × 64")
d.metric("Dataset", f"{len(frames):,} frames")

st.caption(
    "All three pipelines use the same checkpoint. The correction policy is the only rollout variable."
)

for key, default in {
    "results": None,
    "metrics": None,
    "summary": None,
    "run_config": None,
    "current_frame": 1,
    "playing": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if run_button:
    device = torch.device(device_name)
    real = frames[int(start):int(start) + int(horizon) + 1]
    action_sequence = actions[int(start):int(start) + int(horizon)]

    results = {}
    progress = st.progress(0)

    for index, pipeline in enumerate(PIPELINES):
        results[pipeline] = generate_rollout(
            model=model,
            real_frames=real,
            actions=action_sequence,
            strategy=pipeline,
            fixed_interval=int(fixed_interval),
            adaptive_threshold=float(adaptive_threshold),
            device=device,
        )
        progress.progress(int(100 * (index + 1) / len(PIPELINES)))

    progress.empty()

    metric_frames = [
        per_frame_metrics(
            results[pipeline]["real"],
            results[pipeline]["prediction"],
            pipeline,
        )
        for pipeline in PIPELINES
    ]
    metrics = pd.concat(metric_frames, ignore_index=True)

    summary_rows = []
    for pipeline in PIPELINES:
        row = sequence_summary(
            results[pipeline]["real"],
            results[pipeline]["prediction"],
            results[pipeline]["corrections"],
            results[pipeline]["runtime_seconds"],
        )
        summary_rows.append({"Pipeline": LABELS[pipeline], **row})

    st.session_state.results = results
    st.session_state.metrics = metrics
    st.session_state.summary = pd.DataFrame(summary_rows)
    st.session_state.run_config = {
        "start": int(start),
        "horizon": int(horizon),
        "fixed_interval": int(fixed_interval),
        "adaptive_threshold": float(adaptive_threshold),
    }
    st.session_state.current_frame = 1
    st.session_state.playing = False


if st.session_state.results is None:
    st.info("Choose settings and click **Run comparison**.")

    if Path(FINAL_SUMMARY_CSV).exists():
        st.subheader("Frozen five-window experiment")
        st.dataframe(pd.read_csv(FINAL_SUMMARY_CSV), use_container_width=True, hide_index=True)
    st.stop()

results = st.session_state.results
metrics = st.session_state.metrics
summary = st.session_state.summary
run_config = st.session_state.run_config
active_horizon = run_config["horizon"]

st.divider()
st.subheader("Current rollout summary")
st.dataframe(
    summary.style.format(
        {
            "MSE": "{:.6f}",
            "MAE": "{:.6f}",
            "Mean PSNR": "{:.3f}",
            "Mean SSIM": "{:.4f}",
            "Motion Ratio": "{:.3f}x",
            "Ghost Brightness": "{:.5f}",
            "Ghost >0.10": "{:.2%}",
            "Correction Rate": "{:.1%}",
            "Runtime (s)": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Synchronized frame comparison")
controls = st.columns([1, 1, 1, 1, 3])
with controls[0]:
    if st.button("⏮ Previous", use_container_width=True):
        st.session_state.playing = False
        st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
with controls[1]:
    if st.button("▶ Play", use_container_width=True):
        st.session_state.playing = True
with controls[2]:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.playing = False
with controls[3]:
    if st.button("Next ⏭", use_container_width=True):
        st.session_state.playing = False
        st.session_state.current_frame = min(active_horizon, st.session_state.current_frame + 1)
with controls[4]:
    selected = st.slider(
        "Frame",
        min_value=0,
        max_value=active_horizon,
        value=int(st.session_state.current_frame),
        step=1,
    )

if not st.session_state.playing:
    st.session_state.current_frame = int(selected)


def render_player(frame_index):
    st.caption(
        f"Frame {frame_index}/{active_horizon} • dataset index "
        f"{run_config['start'] + frame_index} • playback {playback_speed}×"
    )

    columns = st.columns(4)
    with columns[0]:
        st.markdown("#### Ground Truth")
        st.image(results["baseline"]["real"][frame_index], clamp=True, use_container_width=True)

    for column, pipeline in zip(columns[1:], PIPELINES):
        with column:
            st.markdown(f"#### {LABELS[pipeline]}")
            st.image(results[pipeline]["prediction"][frame_index], clamp=True, use_container_width=True)
            if frame_index in results[pipeline]["corrections"]:
                st.warning("Observation re-anchor for the next step")
            elif pipeline == "baseline":
                st.caption("Open-loop")
            else:
                st.caption("No correction here")


if hasattr(st, "fragment"):
    refresh_seconds = max(0.05, 1.0 / (10.0 * playback_speed))

    @st.fragment(run_every=refresh_seconds if st.session_state.playing else None)
    def player_fragment():
        if st.session_state.playing:
            if st.session_state.current_frame < active_horizon:
                st.session_state.current_frame += 1
            else:
                st.session_state.playing = False
        render_player(int(st.session_state.current_frame))

    player_fragment()
else:
    render_player(int(st.session_state.current_frame))

current_frame = int(st.session_state.current_frame)

st.divider()
st.subheader("Correction behaviour")
st.plotly_chart(correction_figure(results, active_horizon), use_container_width=True)
ca, cb, cc = st.columns(3)
ca.metric("Baseline corrections", len(results["baseline"]["corrections"]))
cb.metric("Fixed effective corrections", len(results["fixed"]["corrections"]))
cc.metric("Adaptive effective corrections", len(results["adaptive"]["corrections"]))
st.caption(
    "Only corrections that can affect a future prediction are counted. A final-frame re-anchor is not counted."
)

st.divider()
st.subheader("Prediction error over time")
metric = st.selectbox("Metric", ["mse", "mae", "psnr", "ssim"], index=0)
st.plotly_chart(metric_figure(metrics, metric, current_frame), use_container_width=True)

st.subheader("Temporal motion")
st.plotly_chart(motion_figure(results, current_frame), use_container_width=True)
st.caption(
    "Motion ratio = predicted mean frame-to-frame motion ÷ real motion. 1.0× is a magnitude match; values above 1.0× mean exaggerated temporal change."
)

st.divider()
st.subheader("Current-frame absolute error")
real_current = results["baseline"]["real"][current_frame]
columns = st.columns(3)
for column, pipeline in zip(columns, PIPELINES):
    with column:
        heat = np.abs(real_current - results[pipeline]["prediction"][current_frame])
        fig = go.Figure(data=go.Heatmap(z=heat, colorbar=dict(title="|error|")))
        fig.update_layout(
            title=LABELS[pipeline],
            height=330,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Frozen five-window experiment")
if Path(FINAL_SUMMARY_CSV).exists():
    st.caption(
        "Final protocol: starts 0, 500, 1000, 2000, 3000; horizon 100; Fixed interval 10; Adaptive threshold 0.0005."
    )
    st.dataframe(pd.read_csv(FINAL_SUMMARY_CSV), use_container_width=True, hide_index=True)
    if Path(FINAL_WINDOWS_CSV).exists():
        with st.expander("Show all window-level runs"):
            st.dataframe(pd.read_csv(FINAL_WINDOWS_CSV), use_container_width=True, hide_index=True)
else:
    st.info("Run `python -m scripts.evaluate_final64 --device cuda` first.")

st.divider()
csv = metrics.copy()
csv["start"] = run_config["start"]
csv["horizon"] = run_config["horizon"]
csv["fixed_interval"] = run_config["fixed_interval"]
csv["adaptive_threshold"] = run_config["adaptive_threshold"]
st.download_button(
    "Download current rollout metrics CSV",
    data=csv.to_csv(index=False).encode("utf-8"),
    file_name=f"final64_start_{run_config['start']}_h{run_config['horizon']}.csv",
    mime="text/csv",
)

with st.expander("Method and interpretation"):
    st.markdown(
        """
**Baseline** runs fully open-loop after the seed frame.

**Fixed Interval** periodically uses an available real observation as the input to the next prediction.

**Adaptive** compares the current prediction with an available real observation and re-anchors the next input only when prediction MSE exceeds the frozen threshold.

The prediction that triggers a correction remains visible and is still scored. Therefore correction does not rewrite the result. All three strategies use the same learned world model, so the comparison isolates rollout correction policy.

**Limitation:** correction can reduce accumulated drift without removing every one-step visual artefact. Adaptive correction also assumes that an observation is available to measure prediction error.
        """
    )
