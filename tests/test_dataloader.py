from torch.utils.data import DataLoader

from training.dataset import WorldModelDataset


dataset = WorldModelDataset(
    folder="data"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

current_frames, actions, next_frames = next(iter(loader))

print("Current :", current_frames.shape)
print("Actions :", actions.shape)
print("Targets :", next_frames.shape)