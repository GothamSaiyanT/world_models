import os
import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# IMPORTANT: page configuration must happen before any other Streamlit call,
# including @st.cache_resource decorators, for compatibility with Streamlit
# versions that enforce this rule.
st.set_page_config(
    page_title="World Model Pipeline Comparison",
    page_icon="📊",
    layout="wide",
)

from config import ModelConfig
from training.dataset import WorldModelDataset
from utils.rollout_generator import RolloutGenerator


CHECKPOINTS = {
    "baseline": "models/best_world_model.npz",
    "adaptive": "models/best_adaptive_model.pt",
    "fixed_interval": "models/best_fixed_interval_model.pt",
}

PIPELINE_LABELS = {
    "baseline": "Baseline",
    "adaptive": "Adaptive",
    "fixed_interval": "Fixed Interval",
}


# ---------------------------------------------------------------------
# Existing project model loading logic, copied here so NO existing file
# in your project has to be edited.
# ---------------------------------------------------------------------
def load_model(pipeline, checkpoint, device):
    if pipeline == "baseline":
        from core.world_model import WorldModel

        model = WorldModel(
            latent_size=128,
            hidden_size=128,
            image_size=64,
        )

        result = model.load(checkpoint)

        if isinstance(result, tuple):
            epoch, best_loss = result
            print(f"[{pipeline}] Checkpoint epoch: {epoch}, best loss: {best_loss}")

    else:
        from core_nn.world_model import WorldModel

        config = ModelConfig()
        model = WorldModel(config)

        checkpoint_data = torch.load(
            checkpoint,
            map_location=device,
        )

        if (
            isinstance(checkpoint_data, dict)
            and "model_state_dict" in checkpoint_data
        ):
            state_dict = checkpoint_data["model_state_dict"]

            if "model_config" in checkpoint_data:
                saved_config = ModelConfig(
                    **checkpoint_data["model_config"]
                )

                if saved_config != config:
                    print(
                        f"[{pipeline}] Rebuilding model using "
                        "the checkpoint's saved configuration."
                    )
                    model = WorldModel(saved_config)

        else:
            state_dict = checkpoint_data

        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading trained models...")
def load_all_models(device_name):
    device = torch.device(device_name)
    models = {}

    for pipeline, checkpoint in CHECKPOINTS.items():
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        models[pipeline] = load_model(
            pipeline,
            checkpoint,
            device,
        )

    return models


@st.cache_resource(show_spinner=False)
def load_dataset():
    return WorldModelDataset(folder="data")


def prepare_sequence(dataset, start, horizon, device):
    initial_frame = (
        dataset[start][0]
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    actions = torch.stack(
        [
            torch.as_tensor(dataset[start + i][1])
            for i in range(horizon)
        ]
    ).to(
        device=device,
        dtype=torch.long,
    ).reshape(-1)

    real_sequence = torch.stack(
        [
            dataset[start + i][0]
            for i in range(horizon + 1)
        ]
    ).detach().cpu()

    return initial_frame, actions, real_sequence


@torch.no_grad()
def generate_prediction(model, initial_frame, actions):
    rollout = RolloutGenerator(model)

    start_time = time.perf_counter()

    predicted = rollout.generate(
        initial_frame=initial_frame,
        actions=actions,
    )

    elapsed = time.perf_counter() - start_time

    if not isinstance(predicted, torch.Tensor):
        predicted = torch.as_tensor(predicted)

    return predicted.detach().cpu(), elapsed


def normalise_sequence_shape(sequence):
    """
    Converts common rollout outputs into [T, C, H, W] where possible.
    """
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.as_tensor(sequence)

    sequence = sequence.detach().cpu()

    # [T, 1, C, H, W] -> [T, C, H, W]
    if sequence.ndim == 5 and sequence.shape[1] == 1:
        sequence = sequence[:, 0]

    # [1, T, C, H, W] -> [T, C, H, W]
    elif sequence.ndim == 5 and sequence.shape[0] == 1:
        sequence = sequence[0]

    return sequence


def align_sequences(real_sequence, predicted_sequence):
    real_sequence = normalise_sequence_shape(real_sequence)
    predicted_sequence = normalise_sequence_shape(predicted_sequence)

    length = min(
        len(real_sequence),
        len(predicted_sequence),
    )

    return (
        real_sequence[:length],
        predicted_sequence[:length],
    )


def tensor_to_image(tensor):
    image = tensor.detach().cpu().float()

    if image.ndim == 4:
        image = image[0]

    if image.ndim == 2:
        arr = image.numpy()
    elif image.ndim == 3:
        if image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)

        arr = image.numpy()
    else:
        raise ValueError(
            f"Unsupported frame shape for display: {tuple(image.shape)}"
        )

    # Gracefully handle either [0, 1] or [0, 255] image ranges.
    if arr.max() > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def ensure_same_shape(real, predicted):
    real = real.float()
    predicted = predicted.float()

    if real.ndim == 4 and real.shape[0] == 1:
        real = real[0]

    if predicted.ndim == 4 and predicted.shape[0] == 1:
        predicted = predicted[0]

    if real.shape != predicted.shape:
        raise ValueError(
            "Real and predicted frame shapes do not match: "
            f"{tuple(real.shape)} vs {tuple(predicted.shape)}"
        )

    return real, predicted


def frame_mse(real, predicted):
    real, predicted = ensure_same_shape(real, predicted)
    return torch.mean((real - predicted) ** 2).item()


def frame_mae(real, predicted):
    real, predicted = ensure_same_shape(real, predicted)
    return torch.mean(torch.abs(real - predicted)).item()


def frame_psnr(real, predicted):
    real, predicted = ensure_same_shape(real, predicted)

    mse = torch.mean((real - predicted) ** 2).item()

    if mse == 0:
        return float("inf")

    maximum = max(
        float(real.max().item()),
        float(predicted.max().item()),
    )

    max_value = 255.0 if maximum > 1.5 else 1.0

    return 10.0 * math.log10(
        (max_value ** 2) / mse
    )


def global_ssim(real, predicted):
    """
    Dependency-free SSIM-style global structural similarity score.

    This is a simple full-frame SSIM calculation rather than a
    windowed skimage implementation, so it keeps the dashboard
    self-contained and avoids requiring another package.
    """
    real, predicted = ensure_same_shape(real, predicted)

    real = real.reshape(-1)
    predicted = predicted.reshape(-1)

    maximum = max(
        float(real.max().item()),
        float(predicted.max().item()),
    )

    data_range = 255.0 if maximum > 1.5 else 1.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = torch.mean(real)
    mu_y = torch.mean(predicted)

    var_x = torch.var(real, unbiased=False)
    var_y = torch.var(predicted, unbiased=False)

    covariance = torch.mean(
        (real - mu_x) * (predicted - mu_y)
    )

    numerator = (
        (2 * mu_x * mu_y + c1)
        * (2 * covariance + c2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + c1)
        * (var_x + var_y + c2)
    )

    if denominator == 0:
        return 1.0

    return float((numerator / denominator).item())


def calculate_pipeline_metrics(
    real_sequence,
    predicted_sequence,
    pipeline,
):
    real_sequence, predicted_sequence = align_sequences(
        real_sequence,
        predicted_sequence,
    )

    rows = []

    for frame_index in range(len(real_sequence)):
        real = real_sequence[frame_index]
        predicted = predicted_sequence[frame_index]

        rows.append(
            {
                "frame": frame_index,
                "pipeline": pipeline,
                "pipeline_label": PIPELINE_LABELS[pipeline],
                "mse": frame_mse(real, predicted),
                "mae": frame_mae(real, predicted),
                "psnr": frame_psnr(real, predicted),
                "ssim": global_ssim(real, predicted),
            }
        )

    return pd.DataFrame(rows)


def find_drift_frame(values, threshold, consecutive):
    count = 0

    for index, value in enumerate(values):
        if float(value) > threshold:
            count += 1

            if count >= consecutive:
                return index - consecutive + 1
        else:
            count = 0

    return None


def error_map(real, predicted):
    real, predicted = ensure_same_shape(real, predicted)

    difference = torch.abs(real - predicted)

    if difference.ndim == 3:
        difference = difference.mean(dim=0)

    return difference.detach().cpu().numpy()


def build_metric_figure(
    metrics_df,
    metric_name,
    threshold,
    current_frame,
):
    figure = go.Figure()

    for pipeline in CHECKPOINTS:
        subset = metrics_df[
            metrics_df["pipeline"] == pipeline
        ]

        figure.add_trace(
            go.Scatter(
                x=subset["frame"],
                y=subset[metric_name],
                mode="lines",
                name=PIPELINE_LABELS[pipeline],
                hovertemplate=(
                    "Frame %{x}<br>"
                    + metric_name.upper()
                    + ": %{y:.6f}<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text="Drift threshold",
        annotation_position="top left",
    )

    figure.add_vline(
        x=current_frame,
        line_dash="dot",
        annotation_text="Current frame",
        annotation_position="top right",
    )

    figure.update_layout(
        title=f"{metric_name.upper()} over rollout",
        xaxis_title="Frame",
        yaxis_title=metric_name.upper(),
        hovermode="x unified",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return figure


def build_summary_table(
    metrics_df,
    runtimes,
    drift_metric,
    threshold,
    consecutive,
):
    rows = []

    for pipeline in CHECKPOINTS:
        subset = metrics_df[
            metrics_df["pipeline"] == pipeline
        ]

        drift_frame = find_drift_frame(
            subset[drift_metric].tolist(),
            threshold,
            consecutive,
        )

        rows.append(
            {
                "Pipeline": PIPELINE_LABELS[pipeline],
                "Mean MSE": subset["mse"].mean(),
                "Mean MAE": subset["mae"].mean(),
                "Mean PSNR": subset["psnr"].replace(
                    [np.inf, -np.inf],
                    np.nan,
                ).mean(),
                "Mean SSIM": subset["ssim"].mean(),
                "First Drift Frame": (
                    drift_frame
                    if drift_frame is not None
                    else "No drift"
                ),
                "Runtime (s)": runtimes[pipeline],
            }
        )

    return pd.DataFrame(rows)


def initialise_state():
    defaults = {
        "comparison_ready": False,
        "results": None,
        "real_sequence": None,
        "metrics": None,
        "runtimes": None,
        "current_frame": 0,
        "playing": False,
        "horizon_used": 0,
        "start_used": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_comparison(
    dataset,
    models,
    start,
    horizon,
    device,
):
    initial_frame, actions, real_sequence = prepare_sequence(
        dataset,
        start,
        horizon,
        device,
    )

    results = {}
    runtimes = {}
    metric_frames = []

    for pipeline, model in models.items():
        prediction, elapsed = generate_prediction(
            model,
            initial_frame,
            actions,
        )

        prediction = normalise_sequence_shape(prediction)

        results[pipeline] = prediction
        runtimes[pipeline] = elapsed

        metric_frames.append(
            calculate_pipeline_metrics(
                real_sequence,
                prediction,
                pipeline,
            )
        )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    return (
        normalise_sequence_shape(real_sequence),
        results,
        runtimes,
        metrics,
    )


# ---------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------

initialise_state()

st.title("World Model Pipeline Comparison")
st.success("Dashboard loaded successfully. Configure the evaluation in the sidebar.")
st.caption(
    "Synchronized comparison of Baseline, Adaptive and Fixed-Interval "
    "rollouts using the same initial frame and action sequence."
)

device_name = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
device = torch.device(device_name)

try:
    dataset = load_dataset()
except Exception as exc:
    st.error(f"Could not load dataset from data/: {exc}")
    st.stop()

max_start = max(len(dataset) - 2, 0)

with st.sidebar:
    st.header("Evaluation Settings")

    st.write(f"Device: **{device_name.upper()}**")
    st.write(f"Dataset samples: **{len(dataset)}**")

    start = st.number_input(
        "Start frame",
        min_value=0,
        max_value=max_start,
        value=0,
        step=1,
    )

    maximum_horizon = max(
        1,
        min(1000, len(dataset) - int(start) - 1),
    )

    default_horizon = min(
        500,
        maximum_horizon,
    )

    horizon = st.slider(
        "Rollout horizon",
        min_value=1,
        max_value=maximum_horizon,
        value=default_horizon,
        step=1,
    )

    playback_speed = st.select_slider(
        "Playback speed",
        options=[0.25, 0.5, 1.0, 2.0, 4.0],
        value=1.0,
    )

    drift_metric = st.selectbox(
        "Drift metric",
        options=["mse", "mae"],
        index=0,
    )

    threshold = st.number_input(
        "Drift threshold",
        min_value=0.0,
        value=0.03,
        step=0.001,
        format="%.5f",
    )

    consecutive = st.slider(
        "Consecutive frames for drift",
        min_value=1,
        max_value=20,
        value=5,
    )

    run_button = st.button(
        "Run Comparison",
        type="primary",
        use_container_width=True,
    )

if run_button:
    if int(start) + int(horizon) >= len(dataset):
        st.error(
            "The selected start frame and horizon exceed the dataset."
        )
    else:
        try:
            models = load_all_models(device_name)

            with st.spinner(
                "Generating synchronized rollouts for all three pipelines..."
            ):
                (
                    real_sequence,
                    results,
                    runtimes,
                    metrics,
                ) = run_comparison(
                    dataset=dataset,
                    models=models,
                    start=int(start),
                    horizon=int(horizon),
                    device=device,
                )

            st.session_state.real_sequence = real_sequence
            st.session_state.results = results
            st.session_state.runtimes = runtimes
            st.session_state.metrics = metrics
            st.session_state.current_frame = 0
            st.session_state.playing = False
            st.session_state.horizon_used = int(horizon)
            st.session_state.start_used = int(start)
            st.session_state.comparison_ready = True

        except Exception as exc:
            st.exception(exc)
            st.stop()

if not st.session_state.comparison_ready:
    st.info(
        "Choose a start frame and rollout horizon, then click "
        "**Run Comparison**."
    )
    st.stop()


real_sequence = st.session_state.real_sequence
results = st.session_state.results
metrics_df = st.session_state.metrics
runtimes = st.session_state.runtimes

available_lengths = [len(real_sequence)]
available_lengths.extend(
    len(sequence)
    for sequence in results.values()
)

max_frame = max(
    0,
    min(available_lengths) - 1,
)

st.session_state.current_frame = min(
    st.session_state.current_frame,
    max_frame,
)

summary_df = build_summary_table(
    metrics_df=metrics_df,
    runtimes=runtimes,
    drift_metric=drift_metric,
    threshold=threshold,
    consecutive=consecutive,
)

st.subheader("Evaluation Summary")

summary_columns = st.columns(3)

for column, pipeline in zip(
    summary_columns,
    CHECKPOINTS,
):
    subset = metrics_df[
        metrics_df["pipeline"] == pipeline
    ]

    drift_frame = find_drift_frame(
        subset[drift_metric].tolist(),
        threshold,
        consecutive,
    )

    with column:
        st.markdown(
            f"### {PIPELINE_LABELS[pipeline]}"
        )
        st.metric(
            "Mean MSE",
            f"{subset['mse'].mean():.6f}",
        )
        st.metric(
            "Mean SSIM",
            f"{subset['ssim'].mean():.4f}",
        )
        st.metric(
            "First drift",
            (
                f"Frame {drift_frame}"
                if drift_frame is not None
                else "No drift"
            ),
        )
        st.caption(
            f"Rollout runtime: {runtimes[pipeline]:.3f} s"
        )

st.dataframe(
    summary_df.style.format(
        {
            "Mean MSE": "{:.6f}",
            "Mean MAE": "{:.6f}",
            "Mean PSNR": "{:.3f}",
            "Mean SSIM": "{:.4f}",
            "Runtime (s)": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("Synchronized Frame Comparison")

control_columns = st.columns(
    [1, 1, 1, 1, 2]
)

with control_columns[0]:
    if st.button("⏮ Previous", use_container_width=True):
        st.session_state.playing = False
        st.session_state.current_frame = max(
            0,
            st.session_state.current_frame - 1,
        )

with control_columns[1]:
    if st.button("▶ Play", use_container_width=True):
        st.session_state.playing = True

with control_columns[2]:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.playing = False

with control_columns[3]:
    if st.button("Next ⏭", use_container_width=True):
        st.session_state.playing = False
        st.session_state.current_frame = min(
            max_frame,
            st.session_state.current_frame + 1,
        )

with control_columns[4]:
    selected_frame = st.slider(
        "Frame",
        min_value=0,
        max_value=max_frame,
        value=st.session_state.current_frame,
        step=1,
        key="frame_slider",
    )

if (
    not st.session_state.playing
    and selected_frame != st.session_state.current_frame
):
    st.session_state.current_frame = selected_frame


# Modern Streamlit supports fragment auto-refresh.
# If an older Streamlit version is installed, the normal controls still work.
if hasattr(st, "fragment"):

    refresh_seconds = max(
        0.05,
        1.0 / (10.0 * playback_speed),
    )

    @st.fragment(
        run_every=refresh_seconds
        if st.session_state.playing
        else None
    )
    def player():
        if st.session_state.playing:
            if st.session_state.current_frame < max_frame:
                st.session_state.current_frame += 1
            else:
                st.session_state.playing = False

        current = st.session_state.current_frame

        st.caption(
            f"Frame {current} / {max_frame} "
            f"• playback {playback_speed}x "
            f"• dataset index "
            f"{st.session_state.start_used + current}"
        )

        columns = st.columns(4)

        with columns[0]:
            st.markdown("#### Ground Truth")
            st.image(
                tensor_to_image(
                    real_sequence[current]
                ),
                use_container_width=True,
            )

        for column, pipeline in zip(
            columns[1:],
            CHECKPOINTS,
        ):
            with column:
                st.markdown(
                    f"#### {PIPELINE_LABELS[pipeline]}"
                )

                st.image(
                    tensor_to_image(
                        results[pipeline][current]
                    ),
                    use_container_width=True,
                )

                row = metrics_df[
                    (metrics_df["pipeline"] == pipeline)
                    & (metrics_df["frame"] == current)
                ]

                if not row.empty:
                    value = float(
                        row.iloc[0][drift_metric]
                    )

                    if value > threshold:
                        st.error(
                            f"{drift_metric.upper()}: "
                            f"{value:.6f} — above threshold"
                        )
                    else:
                        st.success(
                            f"{drift_metric.upper()}: "
                            f"{value:.6f} — within threshold"
                        )

    player()

else:
    current = st.session_state.current_frame

    st.caption(
        "Your Streamlit version does not support automatic fragment "
        "playback. Frame scrubbing and Previous/Next still work."
    )

    columns = st.columns(4)

    with columns[0]:
        st.markdown("#### Ground Truth")
        st.image(
            tensor_to_image(
                real_sequence[current]
            ),
            use_container_width=True,
        )

    for column, pipeline in zip(
        columns[1:],
        CHECKPOINTS,
    ):
        with column:
            st.markdown(
                f"#### {PIPELINE_LABELS[pipeline]}"
            )
            st.image(
                tensor_to_image(
                    results[pipeline][current]
                ),
                use_container_width=True,
            )

current_frame = st.session_state.current_frame

st.divider()
st.subheader("Prediction Error Over Time")

metric_figure = build_metric_figure(
    metrics_df=metrics_df,
    metric_name=drift_metric,
    threshold=threshold,
    current_frame=current_frame,
)

st.plotly_chart(
    metric_figure,
    use_container_width=True,
)

with st.expander("Show additional metric graphs"):
    for metric_name in ["mse", "mae", "psnr", "ssim"]:
        if metric_name == drift_metric:
            continue

        figure = go.Figure()

        for pipeline in CHECKPOINTS:
            subset = metrics_df[
                metrics_df["pipeline"] == pipeline
            ]

            figure.add_trace(
                go.Scatter(
                    x=subset["frame"],
                    y=subset[metric_name],
                    mode="lines",
                    name=PIPELINE_LABELS[pipeline],
                )
            )

        figure.add_vline(
            x=current_frame,
            line_dash="dot",
        )

        figure.update_layout(
            title=f"{metric_name.upper()} over rollout",
            xaxis_title="Frame",
            yaxis_title=metric_name.upper(),
            height=350,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

st.divider()
st.subheader("Current-Frame Error Heatmaps")

heatmap_columns = st.columns(3)

for column, pipeline in zip(
    heatmap_columns,
    CHECKPOINTS,
):
    with column:
        heat = error_map(
            real_sequence[current_frame],
            results[pipeline][current_frame],
        )

        figure = go.Figure(
            data=go.Heatmap(
                z=heat,
                colorbar=dict(
                    title="Absolute error"
                ),
            )
        )

        figure.update_layout(
            title=PIPELINE_LABELS[pipeline],
            height=330,
            margin=dict(
                l=10,
                r=10,
                t=45,
                b=10,
            ),
            xaxis=dict(showticklabels=False),
            yaxis=dict(
                showticklabels=False,
                autorange="reversed",
            ),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

st.divider()

csv_data = metrics_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Evaluation Metrics CSV",
    data=csv_data,
    file_name=(
        f"pipeline_metrics_start_"
        f"{st.session_state.start_used}_"
        f"horizon_{st.session_state.horizon_used}.csv"
    ),
    mime="text/csv",
)
