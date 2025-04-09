from typing import Optional

import gym
import gym_pogs

from balrog.environments.pogs import POGSWrapper
from balrog.environments.pogs.hard import HardPOGS
from balrog.environments.wrappers import GymV21CompatibilityV0


def make_pogs_env(env_name, task, config, render_mode: Optional[str] = None):
    pogs_kwargs = dict(
        num_nodes=config.envs.pogs_kwargs.num_nodes,
        episode_horizon=config.envs.pogs_kwargs.episode_horizon,
        k_nearest=config.envs.pogs_kwargs.k_nearest,
    )
    env = gym.make(task, **pogs_kwargs)
    env = HardPOGS(env, config.envs.pogs_kwargs.min_distance)
    env = POGSWrapper(env)
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
