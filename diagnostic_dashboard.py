import os
import traceback

import streamlit as st

st.set_page_config(page_title="World Model Dashboard Diagnostic", layout="wide")

st.title("World Model Dashboard Diagnostic")
st.write("If you can see this page, Streamlit itself is rendering correctly.")

project_root = os.getcwd()
st.code(project_root, language=None)

st.subheader("1. Required files")

required_paths = [
    "config.py",
    "data/frames.npy",
    "data/actions.npy",
    "models/best_world_model.npz",
    "models/best_adaptive_model.pt",
    "models/best_fixed_interval_model.pt",
    "training/dataset.py",
    "utils/rollout_generator.py",
]

all_present = True

for path in required_paths:
    exists = os.path.exists(path)
    all_present = all_present and exists
    if exists:
        st.success(f"FOUND: {path}")
    else:
        st.error(f"MISSING: {path}")

st.subheader("2. NumPy dataset check")

try:
    import numpy as np

    frames = np.load("data/frames.npy", mmap_mode="r")
    actions = np.load("data/actions.npy", mmap_mode="r")

    st.success("NumPy files loaded successfully.")
    st.write("frames.npy shape:", frames.shape)
    st.write("frames.npy dtype:", str(frames.dtype))
    st.write("actions.npy shape:", actions.shape)
    st.write("actions.npy dtype:", str(actions.dtype))

except Exception as exc:
    st.error("Failed while opening frames.npy or actions.npy.")
    st.exception(exc)
    st.stop()

st.subheader("3. PyTorch check")

try:
    import torch

    st.success(f"PyTorch imported: {torch.__version__}")
    st.write("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        st.write("GPU:", torch.cuda.get_device_name(0))

except Exception as exc:
    st.error("PyTorch import failed.")
    st.exception(exc)
    st.stop()

st.subheader("4. Project import checks")

try:
    from config import ModelConfig
    st.success("Imported ModelConfig from config.py")
except Exception as exc:
    st.error("Importing ModelConfig failed.")
    st.exception(exc)
    st.stop()

try:
    from training.dataset import WorldModelDataset
    st.success("Imported WorldModelDataset")
except Exception as exc:
    st.error("Importing WorldModelDataset failed.")
    st.exception(exc)
    st.stop()

try:
    from utils.rollout_generator import RolloutGenerator
    st.success("Imported RolloutGenerator")
except Exception as exc:
    st.error("Importing RolloutGenerator failed.")
    st.exception(exc)
    st.stop()

st.subheader("5. Dataset constructor check")

try:
    with st.spinner("Constructing WorldModelDataset(folder='data')..."):
        dataset = WorldModelDataset(folder="data")

    st.success("WorldModelDataset constructed successfully.")
    st.write("Dataset length:", len(dataset))

    if len(dataset) > 0:
        sample = dataset[0]

        st.write("dataset[0] Python type:", str(type(sample)))

        if isinstance(sample, (tuple, list)):
            st.write("dataset[0] items:", len(sample))

            for i, value in enumerate(sample):
                shape = getattr(value, "shape", None)
                dtype = getattr(value, "dtype", None)
                st.write(
                    f"Item {i}: type={type(value)}, shape={shape}, dtype={dtype}"
                )

except Exception as exc:
    st.error("WorldModelDataset(folder='data') failed.")
    st.exception(exc)
    st.code(traceback.format_exc())
    st.stop()

st.subheader("6. Model-class import checks")

try:
    from core.world_model import WorldModel as BaselineWorldModel
    st.success("Imported baseline core.world_model.WorldModel")
except Exception as exc:
    st.error("Baseline WorldModel import failed.")
    st.exception(exc)
    st.stop()

try:
    from core_nn.world_model import WorldModel as NNWorldModel
    st.success("Imported core_nn.world_model.WorldModel")
except Exception as exc:
    st.error("Adaptive/fixed WorldModel import failed.")
    st.exception(exc)
    st.stop()

st.subheader("Diagnostic result")

if all_present:
    st.success(
        "All required paths were found and the basic project imports/dataset "
        "checks completed. If comparison_dashboard.py is still blank, the "
        "failure is later in model loading or rollout generation."
    )
else:
    st.warning(
        "At least one required path is missing. Fix the path shown above first."
    )
