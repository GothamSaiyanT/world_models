import gymnasium as gym
import ale_py

from training.preprocessing import Preprocessor

gym.register_envs(ale_py)


env = gym.make(
    "ALE/Breakout-v5",
    render_mode="rgb_array"
)

frame, info = env.reset()

preprocessor = Preprocessor(image_size=64)

processed = preprocessor.process(frame)

print("Processed shape :", processed.shape)
print("Data type       :", processed.dtype)
print("Minimum value   :", processed.min())
print("Maximum value   :", processed.max())

env.close()