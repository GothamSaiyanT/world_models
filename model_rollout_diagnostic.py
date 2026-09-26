import os
import time
import traceback

import streamlit as st
import torch

from config import ModelConfig
from training.dataset import WorldModelDataset
from utils.rollout_generator import RolloutGenerator


CHECKPOINTS = {
    "baseline": "models/best_world_model.npz",
    "adaptive": "models/best_adaptive_model.pt",
    "fixed_interval": "models/best_fixed_interval_model.pt",
}


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
            st.write(f"{pipeline} checkpoint metadata:", result)

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
                    st.info(
                        f"{pipeline}: rebuilding model using saved config."
                    )
                    model = WorldModel(saved_config)
        else:
            state_dict = checkpoint_data

        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


st.set_page_config(
    page_title="Model + Rollout Diagnostic",
    layout="wide",
)

st.title("World Model — Model & Rollout Diagnostic")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.write("Device:", str(device))

dataset = WorldModelDataset(folder="data")

st.write("Dataset length:", len(dataset))

TEST_HORIZON = 5

initial_frame = (
    dataset[0][0]
    .unsqueeze(0)
    .to(device=device, dtype=torch.float32)
)

actions = torch.stack(
    [
        torch.as_tensor(dataset[i][1])
        for i in range(TEST_HORIZON)
    ]
).to(
    device=device,
    dtype=torch.long,
).reshape(-1)

st.write("Initial frame shape:", tuple(initial_frame.shape))
st.write("Actions shape:", tuple(actions.shape))
st.write("Actions:", actions.detach().cpu().tolist())

st.divider()

for pipeline, checkpoint in CHECKPOINTS.items():
    st.subheader(pipeline)

    if not os.path.exists(checkpoint):
        st.error(f"Checkpoint missing: {checkpoint}")
        continue

    try:
        start = time.perf_counter()
        model = load_model(
            pipeline,
            checkpoint,
            device,
        )
        load_time = time.perf_counter() - start

        st.success(
            f"Model loaded successfully in {load_time:.3f} seconds."
        )

        st.write("Model class:", str(type(model)))

    except Exception as exc:
        st.error(f"{pipeline}: MODEL LOAD FAILED")
        st.exception(exc)
        st.code(traceback.format_exc())
        continue

    try:
        rollout = RolloutGenerator(model)

        start = time.perf_counter()

        with torch.no_grad():
            predicted = rollout.generate(
                initial_frame=initial_frame,
                actions=actions,
            )

        rollout_time = time.perf_counter() - start

        if not isinstance(predicted, torch.Tensor):
            predicted = torch.as_tensor(predicted)

        st.success(
            f"Rollout generated successfully in "
            f"{rollout_time:.3f} seconds."
        )

        st.write(
            "Predicted output shape:",
            tuple(predicted.shape),
        )

        st.write(
            "Predicted dtype:",
            str(predicted.dtype),
        )

        st.write(
            "Prediction min:",
            float(predicted.detach().cpu().min()),
        )

        st.write(
            "Prediction max:",
            float(predicted.detach().cpu().max()),
        )

    except Exception as exc:
        st.error(f"{pipeline}: ROLLOUT FAILED")
        st.exception(exc)
        st.code(traceback.format_exc())

    st.divider()

st.subheader("How to read this")

st.write(
    "If all three pipelines show both 'Model loaded successfully' "
    "and 'Rollout generated successfully', then the remaining problem "
    "is inside the comparison dashboard's display/metric layer."
)

st.write(
    "If a pipeline fails here, the error shown above identifies the "
    "checkpoint/model/rollout compatibility problem directly."
)
