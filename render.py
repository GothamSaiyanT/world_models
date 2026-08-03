import os
import torch

from core.world_model import WorldModel
from core.rolloutgenerator import RolloutGenerator
from training.dataset import WorldModelDataset
from utils.video import render_comparison_video


def main(
    start=0,
    horizon=100,
    fps=10
):

    checkpoint = "models/best_world_model.npz"

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    dataset = WorldModelDataset(
        folder="data"
    )

    if start < 0:
        raise ValueError(
            "start cannot be negative."
        )

    if start + horizon >= len(dataset):
        raise ValueError(
            f"Requested frames up to index "
            f"{start + horizon}, but the dataset "
            f"contains only {len(dataset)} samples."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Rendering on:", device)

    # Create and load the trained model
    model = WorldModel(
        latent_size=128,
        hidden_size=128,
        image_size=64
    )

    epoch, best_loss = model.load(
        checkpoint
    )

    print("Checkpoint loaded successfully.")
    print("Checkpoint epoch:", epoch)
    print("Checkpoint best loss:", best_loss)

    model.to(device)
    model.eval()

    # Starting real frame
    initial_frame = (
        dataset[start][0]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32
        )
    )

    # Actions used for the imagined rollout
    actions = torch.stack(
        [
            torch.as_tensor(
                dataset[start + index][1]
            )
            for index in range(horizon)
        ]
    ).to(
        device=device,
        dtype=torch.long
    ).reshape(-1)

    # Matching real frames used only for comparison
    real_sequence = torch.stack(
        [
            dataset[start + index][0]
            for index in range(horizon + 1)
        ]
    )

    print("Initial frame shape:", initial_frame.shape)
    print("Actions shape:", actions.shape)
    print("Real sequence shape:", real_sequence.shape)

    # ---------------------------------------
    # Test the first prediction separately
    # ---------------------------------------

    first_action = actions[0].reshape(1)

    hidden = model.init_hidden(
        batch_size=1
    ).to(device)

    with torch.no_grad():

        first_prediction, _ = model(
            initial_frame,
            first_action,
            hidden
        )

    real_next_frame = (
        dataset[start + 1][0]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32
        )
    )

    first_step_mse = torch.mean(
        (
            first_prediction
            - real_next_frame
        ) ** 2
    ).item()

    print("\nFirst-step diagnostic")
    print("---------------------")
    print(f"First-step MSE: {first_step_mse:.6f}")

    print(
        "Initial frame:",
        f"min={initial_frame.min().item():.4f}",
        f"max={initial_frame.max().item():.4f}",
        f"mean={initial_frame.mean().item():.4f}"
    )

    print(
        "Real next frame:",
        f"min={real_next_frame.min().item():.4f}",
        f"max={real_next_frame.max().item():.4f}",
        f"mean={real_next_frame.mean().item():.4f}"
    )

    print(
        "First prediction:",
        f"min={first_prediction.min().item():.4f}",
        f"max={first_prediction.max().item():.4f}",
        f"mean={first_prediction.mean().item():.4f}",
        f"std={first_prediction.std().item():.4f}"
    )

    # ---------------------------------------
    # Autoregressive rollout
    # ---------------------------------------

    rollout = RolloutGenerator(
        model
    )

    predicted_sequence = rollout.generate(
        initial_frame=initial_frame,
        actions=actions
    )

    print(
        "Predicted sequence shape:",
        predicted_sequence.shape
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_path = (
        "outputs/rollout_comparison.mp4"
    )

    render_comparison_video(
        real_sequence,
        predicted_sequence,
        output_path,
        fps=fps
    )

    print(
        f"\nSaved comparison video to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()