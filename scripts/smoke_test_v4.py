import argparse

import torch

from training.v4_loss import balanced_motion_loss
from v3.device import resolve_accelerator
from v4.config import V4Config
from v4.world_model import WorldModelV4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "tpu", "cpu"],
    )
    args = parser.parse_args()

    runtime = resolve_accelerator(args.device)
    device = runtime.device
    model = WorldModelV4(V4Config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = 2
    current = torch.rand(batch, 1, 64, 64, device=device)
    target = torch.rand(batch, 1, 64, 64, device=device)
    actions = torch.tensor([2, 3], device=device)
    hidden = model.init_hidden(batch, device)

    optimizer.zero_grad(set_to_none=True)
    prediction, hidden = model(current, actions, hidden)
    parts = balanced_motion_loss(prediction, current, target)
    parts.total.backward()
    runtime.optimizer_step(optimizer)
    runtime.sync()

    print("Accelerator:", runtime.label)
    print("Device:", device)
    print("Prediction:", tuple(prediction.shape))
    print("Loss:", float(parts.total.detach().cpu()))
    print("Ghost term:", float(parts.ghost.detach().cpu()))
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    assert prediction.shape == (batch, 1, 64, 64)
    assert bool(torch.isfinite(parts.total).detach().cpu())
    print("V4 64x64 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
