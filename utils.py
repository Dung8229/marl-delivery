import numpy as np
import heapq
import math

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def run_astar(map, start, goal):
    n_rows = len(map)
    n_cols = len(map[0])

    open_list = []
    heapq.heappush(open_list, (manhattan(start, goal), 0, start, []))  # (f = g + h, g, position, path)
    visited = set()
    visited.add(start)

    d = {start: 0}

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current == goal:
            if len(path) == 0:
                return 'S', 0  # đã ở đích
            dx, dy = path[0][0] - start[0], path[0][1] - start[1]
            if (dx, dy) == (-1, 0): return 'U', len(path)
            if (dx, dy) == (1, 0): return 'D', len(path)
            if (dx, dy) == (0, -1): return 'L', len(path)
            if (dx, dy) == (0, 1): return 'R', len(path)
            return 'S', len(path)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if 0 <= next_pos[0] < n_rows and 0 <= next_pos[1] < n_cols:
                if map[next_pos[0]][next_pos[1]] == 0 and next_pos not in visited:
                    visited.add(next_pos)
                    new_g = g + 1
                    d[next_pos] = new_g
                    new_f = new_g + manhattan(next_pos, goal)
                    heapq.heappush(open_list, (new_f, new_g, next_pos, path + [next_pos]))

    return 'S', 100000

def get_window(map, center, window_size):
    x, y = center
    half = window_size // 2
    x_min = max(0, x - half)
    x_max = min(len(map), x + half + 1)
    y_min = max(0, y - half)
    y_max = min(len(map[0]), y + half + 1)

    window_map = [row[y_min:y_max] for row in map[x_min:x_max]]
    return window_map, (x_min, y_min)  # window_map và offset của top-left
  
def is_goal_in_window(goal, offset, window_size):
    gx, gy = goal
    x0, y0 = offset
    return x0 <= gx < x0 + window_size and y0 <= gy < y0 + window_size

def find_border_target(start, goal, offset, window_size):
    # Chuyển tất cả về hệ toạ độ gốc (không phải trong window)
    sx, sy = start
    gx, gy = goal

    # vector hướng
    dx = gx - sx
    dy = gy - sy

    # tránh chia cho 0
    if dx == 0: dx = 1e-5
    if dy == 0: dy = 1e-5

    slope = dy / dx

    half = window_size // 2
    border_points = []

    # Kiểm tra giao điểm với 4 cạnh của window centered tại agent
    for bx in [-half, half]:
        by = slope * bx
        border_points.append((round(sx + bx), round(sy + by)))
    for by in [-half, half]:
        bx = by / slope
        border_points.append((round(sx + bx), round(sy + by)))

    # Chọn điểm nằm trong map
    for x, y in border_points:
        if 0 <= x < len(map) and 0 <= y < len(map[0]):
            return (x, y)
    return start  # fallback
