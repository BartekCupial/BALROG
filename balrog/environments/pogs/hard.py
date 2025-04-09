import gym
import networkx as nx


class HardPOGS(gym.Wrapper):
    def __init__(self, env, difficulty: int = 7):
        super().__init__(env)
        self.difficulty = difficulty
        assert self.difficulty <= self.env.num_nodes / 2

        self.env.unwrapped._choose_target_node = self._choose_target_node

    def _choose_target_node(self):
        # Ensure target is not the same as start and is reachable
        nodes = list(self.env.graph.nodes())
        possible_targets = [n for n in nodes if n != self.env.current_node]

        new_target = self.np_random.choice(possible_targets)
        i = 0
        while nx.shortest_path_length(self.env.graph, self.env.current_node, new_target) <= self.difficulty and i < 100:
            new_target = self.np_random.choice(possible_targets)
            i += 1

        return new_target
