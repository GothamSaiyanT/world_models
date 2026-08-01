from training.dataset import WorldModelDataset


dataset = WorldModelDataset(
    folder="data"
)

print("Dataset size:", len(dataset))

current_frame, action, next_frame = dataset[0]

print("Current frame shape :", current_frame.shape)
print("Action              :", action)
print("Next frame shape    :", next_frame.shape)