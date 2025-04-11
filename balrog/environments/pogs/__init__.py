from balrog.environments.pogs.base import POGSWrapper

__all__ = [POGSWrapper]


def get_instruction_prompt(env, task):
    instruction_prompt = f"""
You are an AI agent designed to navigate the Partially Observable Graph Search (POGS) environment. Your goal is to find and reach a target node in a partially observable graph by exploring the graph structure.

The following are the only valid actions you can take in the game:
{list(range(env.unwrapped.num_nodes))}

In a moment I will present you with an observation containing:
- Adjacency list of each visible
- Your current node position
- The target node you need to reach

The graph has {env.unwrapped.num_nodes} nodes and is partially observable, meaning you can only see connections within a k-nearest neighbor radius of your current position. In this episode k={env.unwrapped.k_nearest}, graph={env.unwrapped.undirected}.

Your action should be a single integer representing the label of the node you want to travel to. This node must be directly connected to your current node.

PLAY
""".strip()

    return instruction_prompt.strip()
