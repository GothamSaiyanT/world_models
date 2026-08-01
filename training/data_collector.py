import os
import gymnasium as gym
import ale_py
import numpy as np

from training.preprocessing import Preprocessor

# Register Atari environments
gym.register_envs(ale_py)


class DataCollector:
    """
    Collects gameplay data from Atari Breakout using random actions.
    Saves:
        - frames.npy
        - actions.npy
    """

    def __init__(
        self,
        environment="ALE/Breakout-v5",
        image_size=64,
        save_folder="data"
    ):

        self.environment = environment
        self.save_folder = save_folder

        self.preprocessor = Preprocessor(
            image_size=image_size
        )

    def collect(self, num_steps=1000):

        # Create the data folder if it doesn't exist
        os.makedirs(
            self.save_folder,
            exist_ok=True
        )

        # Create the Atari environment
        env = gym.make(
            self.environment,
            render_mode="rgb_array"
        )

        observation, info = env.reset()

        # Storage for collected data
        frames = []
        actions = []

        print(f"Collecting {num_steps} frames...")

        for step in range(num_steps):

            # Preprocess and store the current frame
            processed_frame = self.preprocessor.process(
                observation
            )

            frames.append(processed_frame)

            # Random action
            action = env.action_space.sample()
            actions.append(action)

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Restart the episode if the game ends
            if terminated or truncated:
                observation, info = env.reset()

            # Progress update every 200 frames
            if (step + 1) % 200 == 0:
                print(
                    f"Collected {step + 1}/{num_steps} frames"
                )

        env.close()

        # Convert lists to NumPy arrays
        frames = np.asarray(
            frames,
            dtype=np.float32
        )

        actions = np.asarray(
            actions,
            dtype=np.int64
        )

        # Save the dataset
        np.save(
            os.path.join(
                self.save_folder,
                "frames.npy"
            ),
            frames
        )

        np.save(
            os.path.join(
                self.save_folder,
                "actions.npy"
            ),
            actions
        )

        print("\nCollection complete!")
        print(f"Frames shape : {frames.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Saved to folder: {self.save_folder}")

        return frames, actions


if __name__ == "__main__":

    collector = DataCollector()

    collector.collect(
        num_steps=100
    )