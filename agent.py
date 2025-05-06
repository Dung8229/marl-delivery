import numpy as np
import heapq
import math
from utils import manhattan, run_astar, get_window, is_goal_in_window, find_border_target

class Agents:

    def __init__(self):
        """
            TODO:
        """
        self.agents = []
        self.n_robots = 0
        self.state = None

    def init_agents(self, state):
        """
            TODO:
        """
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']

    def get_actions(self, state):
        """
            TODO:
        """
        list_actions = ['S', 'L', 'R', 'U', 'D']
        actions = []
        for i in range(self.n_robots):
            move = np.random.randint(0, len(list_actions))
            pkg_act = np.random.randint(0, 3)
            actions.append((list_actions[move], str(pkg_act)))

        return actions
