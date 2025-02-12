from typing import Optional

import gym
import minihack  # NOQA: F401
from nle import nethack
from nle_code_wrapper.utils.utils import get_function_by_name
from nle_code_wrapper.wrappers import NLECodeWrapper, NoProgressFeedback
from nle_progress import NLEProgressWrapper
from nle_utils.wrappers import FinalStatsWrapper, TaskRewardsInfoWrapper

from balrog.environments.code_minihack.language_wrapper import LanguageWrapper
from balrog.environments.nle.auto_more import AutoMore
from balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit, NoProgressAbort

NETHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if id.startswith("NetHack"):
        NETHACK_ENVS.append(id)


def make_nethack_env(env_name, task, config, render_mode: Optional[str] = None):
    nethack_kwargs = config.envs.code_nethack_kwargs
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

    # NetHack options
    options = []
    for option in nethack.NETHACKOPTIONS:
        if option == "autopickup" and not nethack_kwargs.autopickup:
            options.append("!autopickup")
            continue
        options.append(option)

    kwargs = dict(
        observation_keys=observation_keys,
        penalty_step=nethack_kwargs.penalty_step,
        penalty_time=nethack_kwargs.penalty_time,
        penalty_mode=nethack_kwargs.penalty_mode,
        savedir=nethack_kwargs.savedir,
        save_ttyrec_every=nethack_kwargs.save_ttyrec_every,
        actions=nethack.ACTIONS,
        options=options,
    )

    param_mapping = {
        "max_episode_steps": nethack_kwargs.max_episode_steps,
        "character": nethack_kwargs.character,
        "allow_all_yn_questions": nethack_kwargs.allow_all_yn_questions,
        "allow_all_modes": nethack_kwargs.allow_all_modes,
    }

    for param_name, param_value in param_mapping.items():
        if param_value is not None:
            kwargs[param_name] = param_value

    env = gym.make(task, **kwargs)
    env = NoProgressAbort(env)
    env = NLEProgressWrapper(env)
    env = TaskRewardsInfoWrapper(env)
    env = FinalStatsWrapper(env)
    env = AutoMore(env)
    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    strategies = []
    for strategy_name in nethack_kwargs.strategies:
        strategy_func = get_function_by_name(nethack_kwargs.strategies_loc, strategy_name)
        strategies.append(strategy_func)

    panics = []
    for panic_name in nethack_kwargs.panics:
        panic_func = get_function_by_name(nethack_kwargs.panics_loc, panic_name)
        panics.append(panic_func)

    env = NLECodeWrapper(
        env,
        strategies,
        panics,
        max_strategy_steps=nethack_kwargs.max_strategy_steps,
        add_letter_strategies=nethack_kwargs.add_letter_strategies,
        add_direction_strategies=nethack_kwargs.add_direction_strategies,
        add_more_strategy=nethack_kwargs.add_more_strategy,
    )
    env = NoProgressFeedback(env)

    env = LanguageWrapper(env, vlm=vlm)

    return env
