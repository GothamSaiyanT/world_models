from __future__ import annotations

import os

import ale_py
import cv2
import gymnasium as gym
import numpy as np


gym.register_envs(ale_py)


class BreakoutCollectorV3:
    """
    Collect motion-richer 128x128 Breakout observations.

    Changes from V2:
    - crops score/border regions before resize so the ball/paddle occupy a
      larger proportion of the model input;
    - samples LEFT/RIGHT more often than NOOP/FIRE;
    - still stores the exact action used for every transition.
    """

    def __init__(
        self,
        save_folder="data_v3",
        environment="ALE/Breakout-v5",
        image_size=128,
        seed=42,
    ):
        self.save_folder = save_folder
        self.environment = environment
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def preprocess(self, observation):
        # Breakout is 210x160. y=30:200 removes most scoreboard/border while
        # retaining the playfield and paddle area.
        cropped = observation[30:200, :, :]
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.float32) / 255.0

    def _action_probabilities(self, meanings):
        probabilities = np.ones(len(meanings), dtype=np.float64)
        for index, meaning in enumerate(meanings):
            if meaning == "NOOP":
                probabilities[index] = 0.10
            elif meaning == "FIRE":
                probabilities[index] = 0.10
            elif meaning in {"LEFT", "RIGHT"}:
                probabilities[index] = 0.40
            else:
                probabilities[index] = 0.10
        probabilities /= probabilities.sum()
        return probabilities

    def collect(self, num_steps=10000):
        os.makedirs(self.save_folder, exist_ok=True)
        env = gym.make(self.environment, render_mode="rgb_array")
        observation, _ = env.reset(seed=123)
        meanings = list(env.unwrapped.get_action_meanings())
        probabilities = self._action_probabilities(meanings)
        print("Action meanings:", meanings)
        print("Sampling probabilities:", probabilities.tolist())

        frames = []
        actions = []
        for step in range(int(num_steps)):
            frames.append(self.preprocess(observation))
            action = int(self.rng.choice(len(meanings), p=probabilities))
            actions.append(action)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()

            if (step + 1) % 500 == 0:
                print(f"Collected {step + 1}/{num_steps}")

        env.close()
        frames = np.asarray(frames, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        np.save(os.path.join(self.save_folder, "frames.npy"), frames)
        np.save(os.path.join(self.save_folder, "actions.npy"), actions)

        motion = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2))
        print("Frames:", frames.shape)
        print("Actions:", actions.shape)
        print("Mean frame motion:", float(motion.mean()))
        print("Median frame motion:", float(np.median(motion)))
        print("90th percentile motion:", float(np.percentile(motion, 90)))
        return frames, actions
