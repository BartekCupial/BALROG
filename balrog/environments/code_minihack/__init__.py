def get_available_actions(env):
    available_actions = {}
    for strategy in env.get_wrapper_attr("bot").strategies:
        action = strategy.__name__
        description = strategy.__doc__
        available_actions[action] = description

    return available_actions


def get_instruction_prompt(env, task="MiniHack-ExploreMaze-Hard-Mapped-v0"):
    if "corridorbattle" in task.lower():
        goal = "Your goal is to make best use of the dungeon features to effectively defeat a horde of hostile monsters and reach the downstairs located on the end of the second room."
    elif "corridor" in task.lower():
        goal = "Your goal is explore the rooms and reach the downstairs located in one of the randomly generated rooms"
    elif "quest" in task.lower():
        goal = "Your goal is make use of an object laying around for crossing a lava river (this can be any object allowing levitation or freezing), while fighting monsters and navigating rooms or mazes, finally reach downstairs located at the end of the map"
    elif "wod" in task.lower():
        goal = "Your goal is to zap a wand of death to kill a monster and reach downstairs located behind the monster"
    elif "multiroom" in task.lower():
        goal = "Tour goal is to defeat all monster, open all doors and reach downstairs located in the last room"
    elif "lavacross" in task.lower() or "freeze-lava":
        goal = "Tour goal is to make use of an object laying around for crossing a lava river (this can be any object allowing levitation or freezing) and reach downstairs located on the other side of the lava river"
    elif "hidenseek" in task.lower():
        goal = "The goals is to make use of the environment features, avoid being seen by the monster and quickly run towards the downstairs located in one of the corners of the room. The trees and clouds block the line of sight of the player and a random monster (chosen to be more powerful than the agent). The agent, monsters and spells can pass through clouds unobstructed. The agent and monster cannot pass through trees."
    elif "river" in task.lower():
        goal = "Tour goal is to cross a river using boulders and reach the downstairs located on the other side. Boulders, when pushed into water, create a dry land to walk on allowing the agent to cross it"
    else:
        assert False

    available_actions = get_available_actions(env)
    action_strings = ",\n".join(f"{action}: {description}" for action, description in available_actions.items())
    instruction_prompt = f"""
You are an agent playing MiniHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present a history of actions and observations from the game.

Tip: there is no point in outputting the same action over and over if nothing changes.

{goal}

PLAY!
""".strip()

    return instruction_prompt
