from training.dataset import WorldModelDataset
from training.trainer import Trainer
from core.world_model import WorldModel

model = WorldModel()

dataset = WorldModelDataset()

trainer = Trainer(
    model=model,
    dataset=dataset,
    batch_size=4,
    learning_rate=0.001
)

print("Trainer created successfully.")

loss = trainer.train_epoch()

print("Loss:", loss)