"""Fast architecture/environment smoke test before expensive 128x128 experiments."""

import argparse
import gc
import os
import shutil

import torch

from config import ModelConfig
from highres_128 import IMAGE_SIZE


def check_prediction(name, model, device):
    model.to(device)
    model.eval()
    frame = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
    action = torch.tensor([0], dtype=torch.long, device=device)
    hidden = model.init_hidden(batch_size=1, device=device)
    with torch.no_grad():
        prediction, next_hidden = model(frame, action, hidden)
    assert tuple(prediction.shape) == (1, 1, IMAGE_SIZE, IMAGE_SIZE), prediction.shape
    assert torch.isfinite(prediction).all(), f"{name} produced non-finite values"
    print(f"PASS {name}: prediction={tuple(prediction.shape)}, hidden={tuple(next_hidden.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect-frames",
        type=int,
        default=0,
        help="Also test Atari collection by collecting this many frames into smoke_data_128.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    from core.world_model import WorldModel as BaselineWorldModel
    baseline = BaselineWorldModel(latent_size=128, hidden_size=128, image_size=IMAGE_SIZE)
    check_prediction("baseline 128x128", baseline, device)
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from core_nn.world_model import WorldModel as NeuralWorldModel
    neural = NeuralWorldModel(ModelConfig(image_size=IMAGE_SIZE))
    check_prediction("adaptive/fixed architecture 128x128", neural, device)
    del neural
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.collect_frames:
        if args.collect_frames < 20:
            raise ValueError("Use at least 20 frames for the collection smoke test.")
        from training.data_collector import DataCollector
        folder = "smoke_data_128"
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        frames, actions = DataCollector(
            image_size=IMAGE_SIZE, save_folder=folder
        ).collect(num_steps=args.collect_frames)
        assert tuple(frames.shape[1:]) == (IMAGE_SIZE, IMAGE_SIZE)
        assert len(frames) == len(actions)
        print("PASS Atari collection:", frames.shape, actions.shape)

    print("ALL 128x128 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
