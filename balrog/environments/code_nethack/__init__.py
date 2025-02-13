def get_available_actions(env):
    available_actions = {}
    for strategy in env.get_wrapper_attr("bot").strategies:
        action = strategy.__name__
        description = strategy.__doc__
        if not description:
            continue
        available_actions[action] = description

    return available_actions


def get_instruction_prompt(env, task="MiniHack-ExploreMaze-Hard-Mapped-v0"):

    available_actions = get_available_actions(env)
    action_strings = ",\n".join(f"{action}: {description}" for action, description in available_actions.items())
    instruction_prompt = f"""
You are an agent playing MiniHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.
You can also type letter if prompted by the game message. "Are you sure you want to pray? [ynq] (n)".
In a moment I will present a history of actions and observations from the game.

Tip: there is no point in outputting the same action over and over if nothing changes.

Your goal is to get as far as possible in the game.

PLAY!
""".strip()

    return instruction_prompt
