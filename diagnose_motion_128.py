import numpy as np
import torch

from config import ModelConfig
from training.dataset import WorldModelDataset
from utils.rollout_generator import RolloutGenerator


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HORIZON = 100


# --------------------------------------------------
# Dataset motion
# --------------------------------------------------

frames = np.load("data_128/frames.npy")
actions_np = np.load("data_128/actions.npy")

frame_motion = np.abs(
    frames[1:] - frames[:-1]
).mean(axis=(1, 2))

print("\n===== DATASET =====")

print("Frames:", frames.shape)
print("Actions:", actions_np.shape)

print(
    "Mean real frame-to-frame change:",
    float(frame_motion.mean())
)

print(
    "Median real frame-to-frame change:",
    float(np.median(frame_motion))
)

print(
    "90th percentile real motion:",
    float(np.percentile(frame_motion, 90))
)

unique, counts = np.unique(
    actions_np,
    return_counts=True
)

print("\nAction distribution:")

for action, count in zip(unique, counts):
    print(
        f"Action {action}: "
        f"{count} "
        f"({count / len(actions_np) * 100:.2f}%)"
    )


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = WorldModelDataset(
    folder="data_128"
)

initial_frame = (
    dataset[0][0]
    .unsqueeze(0)
    .to(DEVICE)
)

actions = torch.stack(
    [
        dataset[i][1]
        for i in range(HORIZON)
    ]
).to(
    device=DEVICE,
    dtype=torch.long
)

real = torch.stack(
    [
        dataset[i][0]
        for i in range(HORIZON + 1)
    ]
)


# --------------------------------------------------
# Model loader
# --------------------------------------------------

def load_model(pipeline):

    if pipeline == "baseline":

        from core.world_model import WorldModel

        model = WorldModel(
            latent_size=128,
            hidden_size=128,
            image_size=128
        )

        epoch, loss = model.load(
            "models_128/best_world_model.npz"
        )

        print(
            f"\nBaseline checkpoint: "
            f"epoch={epoch}, loss={loss}"
        )

    else:

        from core_nn.world_model import WorldModel

        model = WorldModel(
            ModelConfig(
                image_size=128
            )
        )

        path = (
            "models_128/best_adaptive_model.pt"
            if pipeline == "adaptive"
            else
            "models_128/best_fixed_interval_model.pt"
        )

        state = torch.load(
            path,
            map_location=DEVICE
        )

        if (
            isinstance(state, dict)
            and "model_state_dict" in state
        ):
            state = state[
                "model_state_dict"
            ]

        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Test each model
# --------------------------------------------------

for pipeline in [
    "baseline",
    "adaptive",
    "fixed_interval"
]:

    print(
        f"\n\n===== {pipeline.upper()} ====="
    )

    model = load_model(pipeline)

    prediction = (
        RolloutGenerator(model)
        .generate(
            initial_frame,
            actions
        )
        .cpu()
    )

    # How much does prediction change?
    pred_motion = (
        torch.abs(
            prediction[1:]
            - prediction[:-1]
        )
        .mean(
            dim=(1, 2, 3)
        )
        .numpy()
    )

    real_motion = (
        torch.abs(
            real[1:]
            - real[:-1]
        )
        .mean(
            dim=(1, 2, 3)
        )
        .numpy()
    )

    print(
        "Mean REAL motion:",
        float(real_motion.mean())
    )

    print(
        "Mean PREDICTED motion:",
        float(pred_motion.mean())
    )

    print(
        "Prediction / real motion ratio:",
        float(
            pred_motion.mean()
            /
            max(real_motion.mean(), 1e-12)
        )
    )

    print(
        "\nFirst 20 predicted motion values:"
    )

    print(
        np.round(
            pred_motion[:20],
            7
        )
    )


    # --------------------------------------------------
    # Does changing the action change the prediction?
    # --------------------------------------------------

    action_predictions = []

    with torch.no_grad():

        for action in range(4):

            hidden = model.init_hidden(
                batch_size=1
            ).to(DEVICE)

            output, _ = model(
                initial_frame,
                torch.tensor(
                    [action],
                    device=DEVICE
                ),
                hidden
            )

            action_predictions.append(
                output.cpu()
            )


    differences = []

    for i in range(4):

        for j in range(i + 1, 4):

            diff = (
                torch.abs(
                    action_predictions[i]
                    - action_predictions[j]
                )
                .mean()
                .item()
            )

            differences.append(diff)

    print(
        "\nMean prediction difference "
        "between different actions:"
    )

    print(
        float(np.mean(differences))
    )