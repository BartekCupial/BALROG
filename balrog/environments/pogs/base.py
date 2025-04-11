from collections import defaultdict

import gymnasium as gym
import numpy as np
from PIL import Image


class POGSWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.language_action_space = list(map(str, list(range(self.env.action_space.n))))

    @property
    def max_steps(self):
        return self.env.unwrapped.max_steps

    @property
    def default_action(self):
        return self.get_text_action(self.env.action_space.sample())

    def get_text_action(self, action):
        return self.language_action_space[action]

    def get_feedback(self, reward, distance_to_target=None):
        if reward is None:
            return ""

        if reward == 100.0:
            return "You have reached the target!"
        elif reward == -1.0:
            return "Invalid Move!"

        return ""

    def pogs_process_obs(self, obs, reward):
        text_observation = self.get_text_observation(obs)
        image = self.get_image_observation(obs)

        feedback = self.get_feedback(reward)

        prompt = f"{text_observation}\n{feedback}" if feedback else f"\n{text_observation}"

        obs = defaultdict(lambda: None)

        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        obs["image"] = image

        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.last_obs = obs
        self.total_reward = 0
        self.distance_to_target = None

        return self.pogs_process_obs(obs, None), info

    def step(self, action):
        action_int = self.language_action_space.index(action)
        obs, reward, term, trun, info = self.env.step(action_int)

        self.total_reward += reward

        if term or trun:
            self.progression = max(0, self.total_reward / 100)

        return self.pogs_process_obs(obs, reward), reward, term, trun, info

    def get_stats(self):
        return {"progression": self.progression}

    def get_text_observation(self, obs):
        num_nodes = int(np.sqrt(len(obs["vector"]) - 2))
        adj_matrix = obs["vector"][:-2].reshape(num_nodes, num_nodes)
        description = ""
        for i, row in enumerate(adj_matrix):
            if any(row):
                neigbhors = list(row.nonzero()[0])
                description += f"node: {i}, neighbors: {neigbhors}\n"

        obsv = f"{description}\ncurrent node: {obs['current_node']}, target node: {obs['target_node']}"

        return obsv

    def get_image_observation(self, obs):
        return Image.fromarray(self.env.unwrapped.get_frame()).convert("RGB")
