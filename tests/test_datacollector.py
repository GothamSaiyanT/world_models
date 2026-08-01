import os

from training.data_collector import DataCollector


collector = DataCollector()

frames, actions = collector.collect(
    num_steps=100
)

print("Frames shape :", frames.shape)
print("Actions shape:", actions.shape)

print("frames.npy exists :", os.path.exists("data/frames.npy"))
print("actions.npy exists:", os.path.exists("data/actions.npy"))