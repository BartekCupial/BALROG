from balrog.environments.pogs.base import POGSWrapper

__all__ = [POGSWrapper]


def get_instruction_prompt(env, task):
    instruction_prompt = f"""
You are an AI agent designed to navigate the Partially Observable Graph Search (POGS) environment. Your goal is to find and reach a target node in a partially observable graph by exploring the graph structure.

In a moment I will present you with an observation containing:
- A list of visible edges (connections between nodes)
- Your current node position
- The target node you need to reach

The graph has {env.num_nodes} nodes and is partially observable, meaning you can only see connections within a k-nearest neighbor radius of your current position. In this episode k={env.k_nearest}, graph={env.undirected}.

Your action should be a single integer representing the label of the node you want to travel to. This node must be directly connected to your current node.

PLAY
""".strip()

    return instruction_prompt.strip()
