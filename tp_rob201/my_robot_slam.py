"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np
from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam
from control import potential_field_control
from occupancy_grid import OccupancyGrid
from planner import Planner # J'importe mon Planner A*
from frontier_exploration import FrontierBasedExploration # Commenté

class MyRobotSlam(RobotAbstract):
    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        super().__init__(lidar_params=lidar_params, odometer_params=odometer_params)

        self.counter = 0
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        # J'initialise mon Planner au lieu de l'explorer
        self.planner = Planner(self.occupancy_grid) 
        self.explorer = FrontierBasedExploration(self.occupancy_grid) 

        self.corrected_pose = np.array([0, 0, 0])
        self.initial_pose = np.array([0, 0, 0])
        self.localise_counter = 0

        #self.goal = np.array([0.0, -500.0, 0.0]) # Je fixe un objectif pour l'exploration
        self.goal = None
        self.path = None
        self.path_index = 0
        self.replan_interval = 20
        self.waypoint_tolerance = 20.0
        self.goal_tolerance = 25.0
        self.frontier_speed_multiplier = 1.3
        
        self.exploration_state = "EXPLORING" 
        self.exploration_iterations = 0
        self.max_exploration_iterations = 700 # Temps réduit pour l'exemple
        self.final_goal = np.array([-400.0, 0.0, 0.0]) 

   
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _follow_path(self, lidar_data, pose: np.ndarray) -> dict:
        if self.goal is None:
            return {"forward": 0.3, "rotation": 0.25}

        dx_g = float(self.goal[0] - pose[0])
        dy_g = float(self.goal[1] - pose[1])
        if np.hypot(dx_g, dy_g) < self.goal_tolerance:
            return {"forward": 0.0, "rotation": 0.0}

        if not self.path or len(self.path) < 2:
            command = potential_field_control(lidar_data, pose, self.goal)
            return self._boost_frontier_command(command)

        while self.path_index < len(self.path) - 1:
            wp = self.path[self.path_index]
            if np.hypot(float(wp[0] - pose[0]), float(wp[1] - pose[1])) > self.waypoint_tolerance:
                break
            self.path_index += 1

        wp = self.path[min(self.path_index, len(self.path) - 1)]
        command = potential_field_control(lidar_data, pose, wp)
        return self._boost_frontier_command(command)

    def _boost_frontier_command(self, command: dict) -> dict:
        if self.goal is None:
            return command
        return {
            "forward": float(np.clip(command["forward"] * self.frontier_speed_multiplier, -2.0, 2.0)),
            "rotation": float(np.clip(command["rotation"] * 1.05, -0.6, 0.6)),
        }

    def control(self):
        raw_pose = self.odometer_values()
        lidar_data = self.lidar()

        if self.localise_counter >= 10: 
            self.tiny_slam.localise(lidar_data, raw_pose)
        self.corrected_pose = np.asarray(self.tiny_slam.get_corrected_pose(raw_pose), dtype=float)
        
        if self.localise_counter == 0:
            self.initial_pose = np.array(self.corrected_pose, dtype=float)

        self.tiny_slam.update_map(lidar_data, self.corrected_pose)
        
        if self.exploration_state == "EXPLORING":
            self._handle_exploration_state(lidar_data)
            self.exploration_iterations += 1
            if self.exploration_iterations >= self.max_exploration_iterations:
                self.exploration_state = "RETURNING_TO_START"
                self.goal = np.array(self.initial_pose, dtype=float)
                self.path = None
                self.path_index = 0
        
        elif self.exploration_state == "RETURNING_TO_START":
            self._handle_returning_to_start_state(lidar_data)
            dist_to_start = float(np.hypot(self.goal[0] - self.corrected_pose[0],
                                           self.goal[1] - self.corrected_pose[1]))
            if dist_to_start < self.goal_tolerance * 2: # 
                self.exploration_state = "FINAL_NAVIGATION"
                self.goal = np.array(self.final_goal, dtype=float) 
                self.path = None
                self.path_index = 0
        
        elif self.exploration_state == "FINAL_NAVIGATION":
            self._handle_final_navigation_state(lidar_data)

        traj = None # Je n'affiche plus la trajectoire de l'explorer, car je n'utilise plus l'explorer pour planifier
        if self.path is not None and len(self.path) >= 2:
            traj = np.asarray(self.path, dtype=float)[:, :2].T

        self.occupancy_grid.display_cv(self.corrected_pose, self.goal, traj)
        self.localise_counter += 1
        return self._follow_path(lidar_data, self.corrected_pose)
    
    # def _handle_exploration_state(self, lidar_data):
    #     # Ici, je n'utilise plus l'explorer pour trouver des frontières
    #     # J'utilise simplement mon Planner pour aller vers mon but fixé
    #     if self.goal is not None and (self.path is None or (self.localise_counter % self.replan_interval == 0)):
    #         self.path = self.planner.plan(self.corrected_pose, self.goal)
    #         self.path_index = 0


    def _handle_exploration_state(self, lidar_data):
        """Handle frontier-based exploration phase."""
        
        # 1. Calcul de la distance et vérification du but atteint
        if self.goal is not None:
            dist_to_goal = float(np.hypot(self.goal[0] - self.corrected_pose[0],
                                          self.goal[1] - self.corrected_pose[1]))
        else:
            dist_to_goal = np.inf

        reached_goal = self.goal is not None and dist_to_goal < self.goal_tolerance

        # 2. Si le but est atteint, on le marque comme visité
        if reached_goal:
            self.explorer.mark_goal_visited(self.goal)

        # 3. Recherche d'une nouvelle frontière si nécessaire
        if self.goal is None or reached_goal:
            next_goal = self.explorer.explore_frontiers(self.corrected_pose)
            if next_goal is not None:
                self.goal = next_goal
                self.path = None
                self.path_index = 0

        # 4. Planification périodique vers l'objectif (A*)
        if self.goal is not None and (self.path is None or (self.localise_counter % self.replan_interval == 0)):
            self.path = self.explorer.plan(self.corrected_pose, self.goal)
            self.path_index = 0
            
            # 5. Gestion d'erreur : si le chemin est impossible, on change de cible
            if self.path is None:
                failed_goal = np.array(self.goal, dtype=float)
                self.explorer.mark_goal_unreachable(self.goal)
                
                next_goal = self.explorer.explore_frontiers(self.corrected_pose)
                if next_goal is not None:
                    self.goal = next_goal
                    self.path = self.explorer.plan(self.corrected_pose, self.goal)
                    self.path_index = 0
                else:
                    # Fallback : on garde l'ancien but pour maintenir le mouvement
                    self.goal = failed_goal
                    self.path_index = 0

   

    def _handle_returning_to_start_state(self, lidar_data):
        if self.path is None or (self.localise_counter % self.replan_interval == 0):
            # J'utilise mon Planner pour le retour
            self.path = self.planner.plan(self.corrected_pose, self.goal)
            self.path_index = 0
    
    def _handle_final_navigation_state(self, lidar_data):
        if self.path is None or (self.localise_counter % self.replan_interval == 0):
            # J'utilise mon Planner pour la navigation finale
            self.path = self.planner.plan(self.corrected_pose, self.final_goal)
            self.path_index = 0
    
    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        current_pose = self.odometer_values()
        self.goal = [-400,40,10]
        
        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), current_pose, self.goal)

        return command
