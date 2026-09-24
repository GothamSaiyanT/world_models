import argparse

import torch

from training.v3_loss import motion_aware_loss
from v3.config import V3Config
from v3.device import resolve_accelerator
from v3.world_model import WorldModelV3


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
    model = WorldModelV3(V3Config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = 2
    current = torch.rand(batch, 1, 128, 128, device=device)
    target = torch.rand(batch, 1, 128, 128, device=device)
    actions = torch.tensor([2, 3], device=device)
    hidden = model.init_hidden(batch, device)

    optimizer.zero_grad(set_to_none=True)
    prediction, hidden = model(current, actions, hidden)
    parts = motion_aware_loss(prediction, current, target)
    parts.total.backward()
    runtime.optimizer_step(optimizer)
    runtime.sync()

    loss_value = float(parts.total.detach().cpu())
    finite = bool(torch.isfinite(parts.total).detach().cpu())

    print("Accelerator:", runtime.label)
    print("Device:", device)
    print("Prediction:", tuple(prediction.shape))
    print("Loss:", loss_value)
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    assert prediction.shape == (batch, 1, 128, 128)
    assert finite
    print("V3 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
