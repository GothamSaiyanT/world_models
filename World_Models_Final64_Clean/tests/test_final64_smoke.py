import torch

from core.loss import StableMotionWeightedMSELoss
from core.world_model import WorldModel


def test_world_model_forward_and_loss():
    model = WorldModel(latent_size=128, hidden_size=128, image_size=64)
    current = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    actions = torch.tensor([2, 3], dtype=torch.long)
    hidden = model.init_hidden(batch_size=2)

    prediction, hidden = model(current, actions, hidden)
    assert prediction.shape == (2, 1, 64, 64)
    assert hidden.shape == (2, 128)

    loss = StableMotionWeightedMSELoss(2.0, 0.05)(prediction, target, current)
    assert torch.isfinite(loss)
    loss.backward()


def test_parameter_count_matches_final_architecture():
    model = WorldModel(latent_size=128, hidden_size=128, image_size=64)
    count = sum(int(p.data.numel()) for p in model.parameters())
    assert count == 10_375_008


def test_dataset_normalizes_uint8(tmp_path):
    import numpy as np
    from training.final64_dataset import SequenceRangeDataset

    frames = np.zeros((20, 64, 64), dtype=np.uint8)
    frames[1] = 148
    actions = np.zeros((20,), dtype=np.int64)
    np.save(tmp_path / "frames.npy", frames)
    np.save(tmp_path / "actions.npy", actions)

    ds = SequenceRangeDataset(tmp_path, start=0, end=20, sequence_length=4)
    sample_frames, sample_actions = ds[0]
    assert sample_frames.dtype == torch.float32
    assert sample_frames.max().item() <= 1.0
    assert abs(sample_frames[1].max().item() - (148.0 / 255.0)) < 1e-6
    assert sample_actions.shape == (4,)
