import subprocess

from nle_code_wrapper.utils.granularity import easy, hard, item, navigation

strategies_easy = [
    *easy,
    *navigation,
    *item,
]

env_tasks = [
    "MiniHack-Corridor-R3-v0",
    "MiniHack-Corridor-R5-v0",
    "CustomMiniHack-Corridor-R8-v0",
    "CustomMiniHack-Corridor-R10-v0",
    "MiniHack-CorridorBattle-v0",
    "MiniHack-CorridorBattle-Dark-v0",
    "MiniHack-WoD-Hard-Full-v0",
    "MiniHack-WoD-Pro-Full-v0",
    "MiniHack-River-v0",
    "MiniHack-River-Monster-v0",
    "MiniHack-River-Lava-v0",
    "MiniHack-River-MonsterLava-v0",
    "MiniHack-River-Narrow-v0",
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
    "MiniHack-Quest-Hard-v0",
    "MiniHack-MultiRoom-N10-v0",
    "MiniHack-MultiRoom-N6-Locked-v0",
    "MiniHack-MultiRoom-N10-Lava-v0",
    "MiniHack-MultiRoom-N6-Monster-v0",
    "MiniHack-MultiRoom-N6-Extreme-v0",
    "MiniHack-MultiRoom-N6-LavaMonsters-v0",
    "MiniHack-Freeze-Lava-Full-v0",  # cross lava freeze
    "MiniHack-LavaCross-Levitate-Full-v0",  # cross lava levitation
    "MiniHack-LavaCross-Full-v0",  # cross lava freeze or levitation
    "MiniHack-HideNSeek-Mapped-v0",
    "MiniHack-HideNSeek-v0",
    "MiniHack-HideNSeek-Lava-v0",
    "MiniHack-HideNSeek-Big-v0",
]

cmd = [
    "python",
    "-m",
    "eval",
    "client.client_name=vllm",
    "client.model_id=meta-llama/Llama-3.1-8B-Instruct",
    f"envs.code_minihack_kwargs.strategies=[{','.join(strategies_easy)}]",
    f"tasks.code_minihack_tasks=[{','.join(env_tasks)}]",
]
subprocess.run(cmd)
