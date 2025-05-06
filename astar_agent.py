import heapq
import math
import copy
from enum import IntEnum, Enum

class Action(Enum):
    PICKUP = '1'
    DROP = '2'
    WAIT = '0'
    
class AStarAgents:
    def __init__(self, window_size=5):
        self.map = None
        self.n_rows = 0
        self.n_cols = 0
        self.window_size = window_size
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.is_init = False
        self.paths = {}  # agent_id -> list of positions by time step

    def init_agents(self, state):
        self.map = state['map']
        self.n_rows = len(self.map)
        self.n_cols = len(self.map[0])

        self.robots = [(robot[0] - 1, robot[1] - 1, 0) for robot in state['robots']]
        self.n_robots = len(self.robots)
        self.robots_target = ['free'] * self.n_robots

        self.packages = [(p[0], p[1] - 1, p[2] - 1, p[3] - 1, p[4] - 1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)

        self.paths = {}

    def update_inner_state(self, state):
        for i in range(len(state['robots'])):
            prev = self.robots[i]
            robot = state['robots'][i]
            self.robots[i] = (robot[0] - 1, robot[1] - 1, robot[2])
            if prev[2] != 0 and self.robots[i][2] == 0:
                delivered_id = prev[2] - 1
                if 0 <= delivered_id < len(self.packages):
                    pkg = self.packages[delivered_id]
                    if pkg is not None and (self.robots[i][0], self.robots[i][1]) == (pkg[3], pkg[4]):
                        self.packages[delivered_id] = None
                        self.packages_free[delivered_id] = False
                self.robots_target[i] = 'free'
            elif self.robots[i][2] != 0:
                self.robots_target[i] = self.robots[i][2]

        self.packages += [(p[0], p[1] - 1, p[2] - 1, p[3] - 1, p[4] - 1, p[5]) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])

    def in_bounds(self, x, y):
        return 0 <= x < self.n_rows and 0 <= y < self.n_cols

    def passable(self, x, y):
        return self.map[x][y] == 0

    def neighbors(self, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny) and self.passable(nx, ny):
                yield (nx, ny)

    def a_star(self, start, goal, constraints):
        frontier = [(0 + self.heuristic(start, goal), 0, start, [])]
        visited = set()

        while frontier:
            f, g, node, path = heapq.heappop(frontier)
            if (node, g) in visited:
                continue
            visited.add((node, g))

            if node == goal:
                print(f"Path found: {path + [node]}")
                return path + [node]

            for neighbor in self.neighbors(*node):
                if neighbor in constraints.get(g + 1, set()):
                    continue
                heapq.heappush(frontier, (g + 1 + self.heuristic(neighbor, goal), g + 1, neighbor, path + [node]))

        return []

    def detect_conflict(self, paths):
        max_len = max(len(p) for p in paths.values())
        for t in range(max_len):
            # Check for vertex conflicts
            positions = {}
            for agent, path in paths.items():
                if t < len(path):
                    pos = path[t]
                else:
                    continue
                if pos in positions:
                    other = positions[pos]
                    print(f"Vertex conflict detected: {agent} and {other} at time {t} at position {pos}")
                    return ("vertex", agent, other, t, pos)
                positions[pos] = agent

            # Check for edge conflicts
            for a1 in paths:
                for a2 in paths:
                    if a1 >= a2:
                        continue
                    p1 = paths[a1]
                    p2 = paths[a2]
                    if t + 1 >= min(len(p1), len(p2)):
                        continue
                    if p1[t] == p2[t + 1] and p1[t + 1] == p2[t]:
                        print(f"Edge conflict detected: {a1} and {a2} at time {t} between {p1[t]} and {p2[t + 1]}")
                        return ("edge", a1, a2, t + 1, p1[t + 1])
            
            # Check for move into static agent        
            for a1 in paths:
                for a2 in paths:
                    if a1 == a2:
                        continue
                    p1 = paths[a1]
                    p2 = paths[a2]
                    if len(p2) == 1:  # a2 is idle
                        if t + 1 < len(p1) and p1[t + 1] == p2[0]:
                            moving_pos = p1[t + 1]
                            idle_pos = p2[0]
                            # Check if the moving agent is adjacent to the idle agent
                            if self.heuristic(idle_pos, moving_pos) == 1:
                                print(f"Idle conflict detected: {a1} moving into idle agent {a2} at time {t + 1} at position {p2[0]}")
                                return ("idle_conflict", a1, a2, t + 1, p2[0])
        return None

    def cbs(self, starts, goals):
        root = {'paths': {}, 'constraints': []}
        for i, (start, goal) in enumerate(zip(starts, goals)):
            path = self.a_star(start, goal, self.build_constraints([], i))
            if not path:
                return {}
            root['paths'][i] = path
        open_list = [root]

        while open_list:
            node = open_list.pop(0)
            conflict = self.detect_conflict(node['paths'])
            if not conflict:
                return node['paths']

            conflict_type, a1, a2, t, pos = conflict
            if conflict_type == "idle_conflict":
                # Handle idle conflict by forcing the idle agent to move
                idle_agent = a2
                moving_agent = a1
                idle_pos = node['paths'][idle_agent][0]

                new_constraints = node['constraints'] + [{
                    'agent': idle_agent,
                    'pos': idle_pos,
                    'time': t
                }]
                new_paths = dict(node['paths'])
                # Find a new path for the idle agent to move away
                neighbors = list(self.neighbors(*idle_pos))
                for neighbor in neighbors:
                    if neighbor != pos:  # Avoid moving to the conflicting position
                        new_path = self.a_star(idle_pos, neighbor, self.build_constraints(new_constraints, idle_agent))
                        if new_path:
                            new_paths[idle_agent] = new_path
                            open_list.append({'paths': new_paths, 'constraints': new_constraints})
                            break
            else:
                for agent in [a1, a2]:
                    new_constraints = node['constraints'] + [{
                        'agent': agent,
                        'pos': pos,
                        'time': t
                    }]
                    new_paths = dict(node['paths'])
                    new_path = self.a_star(starts[agent], goals[agent], self.build_constraints(new_constraints, agent))
                    if not new_path:
                        continue
                    new_paths[agent] = new_path
                    open_list.append({'paths': new_paths, 'constraints': new_constraints})
        return {}
    
    def build_constraints(self, constraints, agent_id):
        result = {}
        for c in constraints:
            if c['agent'] != agent_id:
                continue
            t = c['time']
            if t not in result:
                result[t] = set()
            result[t].add(c['pos'])
        return result


    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
            self.update_inner_state(state)
        else:
            self.update_inner_state(state)

        starts = [(r[0], r[1]) for r in self.robots]
        goals = []

        for i in range(self.n_robots):
            # Check if the robot is already carrying a package
            if self.robots_target[i] != 'free':
                pkg_id = self.robots_target[i] - 1
                # Check if the package is still valid (not delivered)
                if 0 <= pkg_id < len(self.packages) and self.packages[pkg_id] is not None:
                    pkg = self.packages[pkg_id]
                # If the package is not valid, set the target to 'free'
                else:
                    self.robots_target[i] = 'free'
                    # Set the goal to the robot's current position
                    goals.append((self.robots[i][0], self.robots[i][1]))
                    continue
                # If the package is valid, set the goal to the drop-off location
                if self.robots[i][2] == 0:
                    goals.append((pkg[1], pkg[2]))
                else:
                    goals.append((pkg[3], pkg[4]))
            # If the robot is not carrying a package, find the closest package
            else:
                closest_package_id = None
                min_dist = 1e9
                for j, pkg in enumerate(self.packages):
                    if pkg is None or not self.packages_free[j]:
                        continue
                    d = self.heuristic((self.robots[i][0], self.robots[i][1]), (pkg[1], pkg[2]))
                    if d < min_dist:
                        min_dist = d
                        closest_package_id = j
                if closest_package_id is not None:
                    self.packages_free[closest_package_id] = False
                    self.robots_target[i] = self.packages[closest_package_id][0]
                    pkg = self.packages[closest_package_id]
                    goals.append((pkg[1], pkg[2]))
                else:
                    # No packages available, set the goal to the robot's current position
                    goals.append((self.robots[i][0], self.robots[i][1]))

        # Find paths for all robots
        paths = self.cbs(starts, goals)
        actions = []
        for i in range(self.n_robots):
            path = paths.get(i, [starts[i]])
            current = starts[i]
            next_pos = path[1] if len(path) > 1 else current
            move = self.get_direction(current, next_pos)

            action = Action.WAIT.value
            if self.robots_target[i] != 'free' and current == goals[i]:
                if self.robots[i][2] == 0:
                    # Tại điểm lấy hàng, chưa mang hàng → nhặt hàng
                    action = Action.PICKUP.value
                else:
                    # Tại điểm giao hàng, đang mang hàng → đặt hàng
                    action = Action.DROP.value
                    
            # Reset target if the action is invalid
            if action == Action.PICKUP.value and (self.robots_target[i] == 'free' or pkg is None):
                self.robots_target[i] = 'free'
                action = Action.WAIT.value
            elif action == Action.DROP.value and self.robots[i][2] == 0:
                self.robots_target[i] = 'free'
                action = Action.WAIT.value

            actions.append((move, action))

        return actions

    def get_direction(self, from_pos, to_pos):
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        return {(0, 1): 'R', (0, -1): 'L', (1, 0): 'D', (-1, 0): 'U'}.get((dx, dy), 'S')