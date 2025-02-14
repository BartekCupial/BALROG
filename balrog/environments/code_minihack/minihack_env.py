from typing import Optional

import gym
import minihack  # NOQA: F401
import nle_code_wrapper.envs.minihack.envs  # noqa: E402
from nle.env.base import FULL_ACTIONS
from nle_code_wrapper.utils.utils import get_function_by_name
from nle_code_wrapper.wrappers import NLECodeWrapper, NoProgressFeedback, SaveOnException

from balrog.environments.code_minihack.language_wrapper import LanguageWrapper
from balrog.environments.nle.auto_more import AutoMore
from balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit, NoProgressAbort

MINIHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if id.split("-")[0] == "MiniHack":
        MINIHACK_ENVS.append(id)


CUSTOM_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if id.split("-")[0] == "CustomMiniHack":
        CUSTOM_ENVS.append(id)


def make_minihack_env(env_name, task, config, render_mode: Optional[str] = None):
    minihack_kwargs = config.envs.code_minihack_kwargs
    vlm = True if config.agent.max_image_history > 0 else False

    observation_keys = (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        "glyphs",
        "inv_glyphs",
        "inv_strs",
        "inv_letters",
        "inv_oclasses",
    )

    kwargs = dict(
        observation_keys=observation_keys,
        penalty_step=minihack_kwargs.penalty_step,
        penalty_time=minihack_kwargs.penalty_time,
        penalty_mode=minihack_kwargs.penalty_mode,
        savedir=minihack_kwargs.savedir,
        save_ttyrec_every=minihack_kwargs.save_ttyrec_every,
        actions=FULL_ACTIONS,
    )

    param_mapping = {
        "max_episode_steps": minihack_kwargs.max_episode_steps,
        "character": minihack_kwargs.character,
        "autopickup": minihack_kwargs.autopickup,
        "allow_all_yn_questions": minihack_kwargs.allow_all_yn_questions,
        "allow_all_modes": minihack_kwargs.allow_all_modes,
    }

    for param_name, param_value in param_mapping.items():
        if param_value is not None:
            kwargs[param_name] = param_value

    env = gym.make(task, **kwargs)
    env = NoProgressAbort(env)
    env = AutoMore(env)
    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    strategies = []
    for strategy_name in minihack_kwargs.strategies:
        strategy_func = get_function_by_name(minihack_kwargs.strategies_loc, strategy_name)
        strategies.append(strategy_func)

    panics = []
    for panic_name in minihack_kwargs.panics:
        panic_func = get_function_by_name(minihack_kwargs.panics_loc, panic_name)
        panics.append(panic_func)

    env = NLECodeWrapper(
        env,
        strategies,
        panics,
        max_strategy_steps=minihack_kwargs.max_strategy_steps,
        add_letter_strategies=minihack_kwargs.add_letter_strategies,
        add_direction_strategies=minihack_kwargs.add_direction_strategies,
        add_more_strategy=minihack_kwargs.add_more_strategy,
    )
    env = NoProgressFeedback(env)

    # would be best to save in run dir, but we don't have access to it from here
    failed_game_path = f"{config.eval.output_dir}/failed_games"
    env = SaveOnException(env, failed_game_path=failed_game_path)

    env = LanguageWrapper(env, vlm=vlm)

    return env
