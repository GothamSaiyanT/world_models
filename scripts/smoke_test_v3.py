import torch

from training.v3_loss import motion_aware_loss
from v3.config import V3Config
from v3.world_model import WorldModelV3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldModelV3(V3Config()).to(device)
    batch = 2
    current = torch.rand(batch, 1, 128, 128, device=device)
    target = torch.rand(batch, 1, 128, 128, device=device)
    actions = torch.tensor([2, 3], device=device)
    hidden = model.init_hidden(batch, device)
    prediction, hidden = model(current, actions, hidden)
    parts = motion_aware_loss(prediction, current, target)
    parts.total.backward()
    print("Device:", device)
    print("Prediction:", tuple(prediction.shape))
    print("Loss:", float(parts.total.detach().cpu()))
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    assert prediction.shape == (batch, 1, 128, 128)
    assert torch.isfinite(parts.total)
    print("V3 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
