"""
Planner class
Implementation of A*
"""

import heapq
import math
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None
        self.free_threshold = -0.5
        # Obstacle inflation parameters
        self.inflation_radius_m = 0.5
        self.inflation_cells = None

    def get_neighbors(self, current_cell):
        """Return list of free (i.e. not obstacle) neighbour cells 
        with the format of current_cell: (i, j) in the map frame
        """
        neighbor_list = []
        i, j = current_cell
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        if self.map_walls is None:
            return neighbor_list
            
        x_max, y_max = self.map_walls.shape
        for dx, dy in deltas:
            ni, nj = i + dx, j + dy
            if 0 <= ni < x_max and 0 <= nj < y_max:
                # free cell when map_walls is False (or 0)
                if not bool(self.map_walls[ni, nj]):
                    neighbor_list.append((ni, nj))
        
        return neighbor_list

    def heuristic(self, cell_1: Tuple[int, int], cell_2: Tuple[int, int]):
        """Return heuristic goal distance (Euclidean distance between cells)"""
        dx = cell_2[0] - cell_1[0]
        dy = cell_2[1] - cell_1[1]
        h = math.hypot(dx, dy)
        return h

    def reconstruct_path(self, came_from, goal):
        """Extract path after cost computation
        Returns list of poses [x,y,0] in world coordinates
        """
        total_path = [goal]
        cell = goal
        while cell in came_from.keys():
            cell = came_from[cell]
            total_path.insert(0, cell)

        total_path = np.array(total_path)
        # Convert map coordinates to world coordinates (center of cells)
        traj_world_x, traj_world_y = self.grid.conv_map_to_world(
            total_path[:, 0] , 
            total_path[:, 1] 
        )
        # Return as list of poses [x, y, 0]
        return np.vstack((traj_world_x, traj_world_y, np.zeros_like(traj_world_x))).T.tolist() #transforma uma matri de dados bruta o formato exato que o código espera
        #ele retorna uma matriz [[x1, x2, 0], [y1, y2, 0], [0, 0, 0]] e a gente quer [[x1, y1, 0], [x2, y2, 0]] e o tolist converte essa matriz em uma lista de listas
    def plan(self, start, goal):
        """
        Compute a path using A* and return a list of poses [x,y,0].
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # Convert to map coordinates
        start: Tuple[int, int] = self.grid.conv_world_to_map(start[0], start[1])
        goal: Tuple[int, int] = self.grid.conv_world_to_map(goal[0], goal[1])

        # Build obstacle mask from occupancy values.
        # Free cells are <= free_threshold, everything else is treated as wall.
        known_free = self.grid.occupancy_map <= self.free_threshold #
        self.map_walls = ~known_free # Invert to get walls (True where there are obstacles, False where free)
        self.inflation_cells = int(math.ceil(self.inflation_radius_m / float(self.grid.resolution))) # number of cells to inflate based on the desired radius and grid resolution
        
        if self.inflation_cells > 0: # If inflation is needed, perform dilation to inflate obstacles
            kernel_size = 2 * self.inflation_cells + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.map_walls = cv2.dilate(self.map_walls.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Check if start or goal are in obstacles
        if self.map_walls[start] or self.map_walls[goal]:
            print('failed getting to objective: start or goal in obstacle')
            return None

        # Min heap to contain values to explore next
        open_set = [(0.0, start)]
        heapq.heapify(open_set)

        # Dictionary to trace back route
        came_from = {}

        # Cost to get to each cell
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0.0

        # Best guess of cost for each cell (cost + heuristic)
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = 0.0 + self.heuristic(start, goal)

        while len(open_set) > 0:
            current_f, current_cell = heapq.heappop(open_set)
            
            # Lazy deletion: skip stale entries
            if current_f > f_score[current_cell]:
                continue
            
            if current_cell == goal:
                return self.reconstruct_path(came_from, goal)

            neighbours = self.get_neighbors(current_cell)
            for cell in neighbours:
                tentative_g_score = g_score[current_cell] + self.heuristic(current_cell, cell)
                if tentative_g_score < g_score[cell]:
                    # Better path, recording it
                    came_from[cell] = current_cell
                    g_score[cell] = tentative_g_score
                    f_score[cell] = tentative_g_score + self.heuristic(cell, goal)
                    heapq.heappush(open_set, (f_score[cell], cell))

        # Goal was never reached
        print('failed getting to objective')
        return None

    def explore_frontiers(self):
        """Frontier based exploration"""
        goal = np.array([0, 0, 0])
        return goal
