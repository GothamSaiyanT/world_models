import argparse
import torch

from config_final64 import (
    BALL_BRIGHTNESS_THRESHOLD,
    BALL_PADDLE_BAND_PX,
    BALL_WALL_MARGIN_PX,
    BALL_WEIGHT,
    HIDDEN_SIZE,
    IMAGE_SIZE,
    LATENT_SIZE,
    MOTION_THRESHOLD,
    MOTION_WEIGHT,
)
from core.loss import BallAwareMotionWeightedMSELoss
from core.world_model import WorldModel
from training.optimizer import Adam
from training.trainer import clip_gradient_norm


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)

    model = WorldModel(
        latent_size=LATENT_SIZE,
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
    ).to(device)

    params = list(model.parameters())
    optimizer = Adam(params, learning_rate=0.001)
    criterion = BallAwareMotionWeightedMSELoss(
        motion_weight=MOTION_WEIGHT,
        motion_threshold=MOTION_THRESHOLD,
        ball_weight=BALL_WEIGHT,
        wall_margin=BALL_WALL_MARGIN_PX,
        paddle_band=BALL_PADDLE_BAND_PX,
        ball_brightness_threshold=BALL_BRIGHTNESS_THRESHOLD,
    )

    batch = 2
    current = torch.rand(batch, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
    target = torch.rand(batch, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
    action = torch.tensor([2, 3], dtype=torch.long, device=device)
    hidden = model.init_hidden(batch_size=batch, device=device)

    prediction, hidden = model(current, action, hidden)
    loss = criterion(prediction, target, current)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_gradient_norm(params, max_norm=1.0)
    optimizer.step()

    count = sum(int(p.data.numel()) for p in params)

    print("Accelerator:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("Device:", device)
    print("Prediction:", tuple(prediction.shape))
    print("Loss:", float(loss.item()))
    print("Gradient norm:", float(grad_norm))
    print("Parameters:", f"{count:,}")
    print("FINAL64 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
