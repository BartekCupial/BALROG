from typing import Optional

import gym_pogs
import gymnasium as gym

from balrog.environments.pogs import POGSWrapper


def make_pogs_env(env_name, task, config, render_mode: Optional[str] = None):
    pogs_kwargs = dict(
        num_nodes=config.envs.pogs_kwargs.num_nodes,
        k_nearest=config.envs.pogs_kwargs.k_nearest,
        min_backtracks=config.envs.pogs_kwargs.min_backtracks,
        max_episode_steps=config.envs.pogs_kwargs.max_episode_steps,
    )
    env = gym.make(task, **pogs_kwargs, render_mode=render_mode)
    env = POGSWrapper(env)

    return env
