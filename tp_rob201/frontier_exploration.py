"""Frontier-based exploration helpers for occupancy-grid SLAM."""

import cv2
import numpy as np

from occupancy_grid import OccupancyGrid
from planner import Planner


class FrontierBasedExploration:
    """Multi-objective frontier-based exploration based on Fang 2013 and Da Silva 2020.
    
    Handles frontier detection, selection via multi-objective utility function,
    and path planning using centralized Planner class.
    """

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.planner = Planner(occupancy_grid)  # Use centralized A* planner
        
        # Grid thresholds for frontier detection
        self.free_threshold = -0.1 # Cells with occupancy <= this are considered free
        self.unknown_threshold = 0.1 # Cells with occupancy between -0.1 and 0.1 are considered unknown
        
        # Frontier tracking to avoid revisitation
        self.visited_goals = []
        self.unreachable_goals = []
        self.revisit_radius = 50.0
        self.unreachable_penalty_radius = 1.0
        
        # Multi-objective weights (Cogent Frontier Exploration inspired)
        self.k_cluster = 1.0      # Cluster size importance
        self.k_distance = 3     # Distance importance (lower = prefer closer)
        self.k_information = 1.2  # Information gain importance
        self.k_clearance = 0.8    # Clearance/reachability importance
        self.k_unreachable = 1.5  # Penalty for unreachable areas

    def mark_goal_visited(self, goal_pose):
        """Remember a goal so we do not immediately select the same frontier again."""
        if goal_pose is None:
            return

        goal_xy = np.asarray(goal_pose[:2], dtype=float) # Only track (x,y) for visited goals
        for visited_goal in self.visited_goals:
            if np.hypot(goal_xy[0] - visited_goal[0], goal_xy[1] - visited_goal[1]) < self.revisit_radius: # Si un but similaire a déjà été visité récemment, ne pas l'ajouter à la liste des buts visités
                return

        self.visited_goals.append(goal_xy) # Ajouter le but actuel à la liste des buts visités
        if len(self.visited_goals) > 50: # Limiter la taille de la liste pour éviter une croissance infinie
            self.visited_goals = self.visited_goals[-50:] # Garder seulement les 50 derniers buts visités
    
    def mark_goal_unreachable(self, goal_pose):
        """Mark a frontier goal as unreachable to avoid repeated attempts."""
        if goal_pose is None:
            return
        goal_xy = np.asarray(goal_pose[:2], dtype=float)
        self.unreachable_goals.append(goal_xy) # Ajouter le but actuel à la liste des buts inaccessibles

    def _known_free_mask(self):
        """Return boolean mask of known free cells in occupancy grid."""
        return self.grid.occupancy_map <= self.free_threshold

    def _frontier_mask(self):
        """Detect frontier cells: free cells adjacent to unknown cells."""
        occupancy = self.grid.occupancy_map
        known_free = occupancy <= self.free_threshold
        unknown = np.abs(occupancy) < self.unknown_threshold

        # Dilate unknown cells to find adjacency
        kernel = np.ones((3, 3), dtype=np.uint8)
        adjacent_to_unknown = cv2.dilate(unknown.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # Frontier = free cells adjacent to unknown
        return known_free & adjacent_to_unknown

    def plan(self, start, goal):
        """Plan path using centralized Planner (A* algorithm).
        
        Delegates to Planner.plan() to avoid code duplication.
        Returns list of poses [x, y, 0] or None if planning fails.
        """
        return self.planner.plan(start, goal)

    def explore_frontiers(self, current_pose=None, min_cluster_size=5):
        """Return the best frontier goal as a world pose [x, y, 0] using multi-objective selection.
        
        Selects frontier based on:
        - Cluster size (prefers larger contiguous frontier regions)
        - Distance (prefers closer frontiers)
        - Information gain estimate
        - Clearance/reachability
        - Avoids recently visited areas
        """
        frontier_mask = self._frontier_mask()
        if not np.any(frontier_mask):
            return None

        n_labels, labels = cv2.connectedComponents(frontier_mask.astype(np.uint8), connectivity=8)
        if n_labels <= 1:
            return None

        current_xy = np.array([0.0, 0.0], dtype=float) if current_pose is None else np.asarray(current_pose[:2], dtype=float)
        best_goal = None
        best_utility = -np.inf  # Higher utility is better

        for label in range(1, n_labels):
            coords = np.argwhere(labels == label)
            if coords.shape[0] < min_cluster_size:
                continue

            # Find representative point of frontier cluster (closest to centroid)
            centroid = coords.mean(axis=0)
            representative = coords[np.argmin(np.sum((coords - centroid) ** 2, axis=1))]
            x_world, y_world = self.grid.conv_map_to_world(representative[0] + 0.5, representative[1] + 0.5)
            candidate = np.array([float(x_world), float(y_world), 0.0], dtype=float)
            
            # Skip if too close to recently visited frontier
            if self._is_too_close_to_visited(candidate):
                continue
            
            # Skip if marked unreachable
            if self._is_marked_unreachable(candidate):
                continue
            
            # Compute multi-objective utility score
            utility = self._compute_utility(candidate, coords, current_xy)
            
            if utility > best_utility:
                best_utility = utility
                best_goal = candidate

        # Fallback pass: if strict filters removed all candidates, retry ignoring
        # visited/unreachable constraints to avoid getting stuck with no goal.
        if best_goal is None:
            for label in range(1, n_labels):
                coords = np.argwhere(labels == label)
                if coords.shape[0] < min_cluster_size:
                    continue

                centroid = coords.mean(axis=0)
                representative = coords[np.argmin(np.sum((coords - centroid) ** 2, axis=1))]
                x_world, y_world = self.grid.conv_map_to_world(representative[0] + 0.5, representative[1] + 0.5)
                candidate = np.array([float(x_world), float(y_world), 0.0], dtype=float)
                utility = self._compute_utility(candidate, coords, current_xy)

                if utility > best_utility:
                    best_utility = utility
                    best_goal = candidate

        return best_goal

    def _is_too_close_to_visited(self, candidate):
        """Check if candidate is too close to previously visited goals."""
        if not self.visited_goals:
            return False
        return any(
            np.hypot(candidate[0] - visited_goal[0], candidate[1] - visited_goal[1]) < self.revisit_radius
            for visited_goal in self.visited_goals
        )

    def _is_marked_unreachable(self, candidate):
        """Check if candidate is in unreachable area."""
        if not self.unreachable_goals:
            return False
        return any(
            np.hypot(candidate[0] - unreachable[0], candidate[1] - unreachable[1]) < self.unreachable_penalty_radius
            for unreachable in self.unreachable_goals
        )

    def _compute_utility(self, candidate, frontier_coords, current_xy):
        """Compute multi-objective utility score for frontier candidate.
        
        Combines: cluster_score, distance_score, information_gain_score, clearance_score
        """
        # 1. Cluster score: prefer larger frontier regions
        cluster_score = float(frontier_coords.shape[0])
        
        # 2. Distance score: prefer closer frontiers (inverted to maximize)
        distance = np.hypot(candidate[0] - current_xy[0], candidate[1] - current_xy[1])
        distance_score = 1.0 / (1.0 + distance)  # Normalized to [0,1]
        
        # 3. Information gain: estimate unknown cells around frontier
        info_gain_score = self._estimate_information_gain(frontier_coords)
        
        # 4. Clearance score: heuristic reachability estimate
        clearance_score = self._estimate_clearance(frontier_coords)
        
        # Weighted combination
        utility = (
            self.k_cluster * cluster_score +
            self.k_distance * distance_score +
            self.k_information * info_gain_score +
            self.k_clearance * clearance_score
        )
        
        return utility

    def _estimate_information_gain(self, frontier_coords):
        """Estimate information gain from exploring this frontier region."""
        # Count nearby unknown cells
        if len(frontier_coords) == 0:
            return 0.0
        
        centroid = frontier_coords.mean(axis=0) # Coordenates of the center of the frontier cluster
        search_radius = 10  # cells 
        
        unknown = np.abs(self.grid.occupancy_map) < self.unknown_threshold # Mask of unknown cells
        x_min = max(0, int(centroid[0] - search_radius))
        x_max = min(unknown.shape[0], int(centroid[0] + search_radius))
        y_min = max(0, int(centroid[1] - search_radius))
        y_max = min(unknown.shape[1], int(centroid[1] + search_radius))
        
        nearby_unknown = np.sum(unknown[x_min:x_max, y_min:y_max]) # Nombre de cellules inconnues dans la zone de recherche autour du centre du frontiere
        max_unknown = search_radius ** 2
        
        return float(nearby_unknown) / max(max_unknown, 1.0)

    def _estimate_clearance(self, frontier_coords):
        """Estimate clearance/reachability of frontier region (heuristic)."""
        if len(frontier_coords) == 0:
            return 0.0
        
        # Count free space around frontier
        free_nearby = 0
        search_radius = 5
        
        for coord in frontier_coords[:min(10, len(frontier_coords))]:  # Sample frontier
            x, y = int(coord[0]), int(coord[1])
            x_min = max(0, x - search_radius)
            x_max = min(self.grid.occupancy_map.shape[0], x + search_radius)
            y_min = max(0, y - search_radius)
            y_max = min(self.grid.occupancy_map.shape[1], y + search_radius)
            
            free_nearby += np.sum(self.grid.occupancy_map[x_min:x_max, y_min:y_max] <= self.free_threshold)
        
        max_free = search_radius ** 2 * min(10, len(frontier_coords))
        return float(free_nearby) / max(max_free, 1.0)
