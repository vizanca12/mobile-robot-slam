""" A simple robotics navigation code including SLAM, exploration, planning"""

import math

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    # def _score(self, lidar, pose):
    #     """
    #     Computes the sum of log probabilities of laser end points in the map
    #     lidar : placebot object with lidar data
    #     pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
    #     """
    #     # TODO for TP4

    #     score = 0

    #     x_r, y_r, theta_r = pose

    #     distances = lidar.get_sensor_values()
    #     angles = lidar.get_ray_angles()

    #     valid_mask = distances < distances.max()  

    #     distances = distances[valid_mask]
    #     angles = angles[valid_mask]

    #     x_world = x_r + distances * np.cos(theta_r + angles)
    #     y_world = y_r + distances * np.sin(theta_r + angles)

    #     x_map , y_map = self.grid.conv_world_to_map(x_world, y_world)

    #     valid_mask = (x_map >=0) & (x_map < self.grid.x_max_map) & (y_map >= 0) & (y_map< self.grid.y_max_map)
    #     x_map = x_map[valid_mask]
    #     y_map= y_map[valid_mask]

    #     score = np.sum(self.grid.occupancy_map[x_map, y_map])

    #     return score
    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates

        essa função representa a corrção da localização do robô, ou seja, a correção da posição do robô no mapa. A função recebe os dados do lidar e a posição atual do robô (pose) e calcula uma pontuação (score) 
        que representa a qualidade da localização do robô com base nos dados do lidar e no mapa de ocupação.
        A função começa obtendo os valores dos sensores do lidar (distâncias) e os ângulos dos raios do lidar. Em seguida, calcula as coordenadas dos pontos finais dos raios do lidar no mundo usando a posição do robô e os dados do lidar.
        """
        # TODO for TP4

        score = 0
        distance = np.asarray(lidar.get_sensor_values(), dtype=float)
        angles = np.asarray(lidar.get_ray_angles(), dtype=float)

        # Filtrar pontos no alcance máximo do laser que é 600
        valid_mask = distance < 600.0

        distance = distance[valid_mask]
        angles = angles[valid_mask]
        # Estimer les positions des d ́etections du laser dans le rep`ere absolu,
        # en utilisant la position du robot et les données du lidar
        x = pose[0] + distance * np.cos(angles + pose[2])
        y = pose[1] + distance * np.sin(angles + pose[2])
  
        #Convertir ces positions en index de cellules dans la grille d’occupation 
        x_idx, y_idx = self.grid.conv_world_to_map(x, y)

        #et supprimer les points hors de la carte
        valid_mask = (x_idx >=0) & (x_idx < self.grid.x_max_map) & (y_idx >= 0) & (y_idx < self.grid.y_max_map)
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        #Calculer la somme des log probabilit ́es des points d’extrémité du laser dans la carte d’occupation
        score = np.sum(self.grid.occupancy_map[x_idx, y_idx])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        x_oref, y_oref, theta_oref = odom_pose_ref

        x0, y0, theta0 = odom_pose

        d0 = math.sqrt(x0**2 + y0**2)
        alpha0 = math.atan2(y0, x0)

        cos = math.cos(theta_oref + alpha0)
        sin = math.sin(theta_oref + alpha0)

        x = x_oref + d0 * cos
        y = y_oref + d0 * sin
        theta = theta_oref + theta0

        corrected_pose = [x, y, theta]

        return corrected_pose

 

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        sigma = np.array([0.05, 0.05, 0.001])  # é usado como desvio padrão em np.random.normal
        current_pose = self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref)

        best_score = self._score(lidar, current_pose)
        best_ref = np.array(self.odom_pose_ref, dtype=float)

        # Arrêt après N tirages consécutifs sans amélioration
        N = 100
        no_improve = 0 

        while no_improve < N:
            offset = np.random.normal(loc=0.0, scale=sigma)
            candidate_ref = best_ref + offset

            candidate_pose = self.get_corrected_pose(raw_odom_pose, candidate_ref)
            candidate_score = self._score(lidar, candidate_pose)

            if candidate_score > best_score:
                best_score = candidate_score
                best_ref = candidate_ref
                no_improve = 0
            else:
                no_improve += 1 # Incrementer le compteur de tirages sans amélioration

        self.odom_pose_ref = best_ref

        return best_score

    # def update_map(self, lidar, pose):
    #     """
    #     Bayesian map update with new observation
    #     lidar : placebot object with lidar data
    #     pose : [x, y, theta] nparray, corrected pose in world coordinates
    #     """
        # TODO for TP4

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        x_r, y_r, theta_r = pose

        distances = np.asarray(lidar.get_sensor_values(), dtype=float)
        angles = np.asarray(lidar.get_ray_angles(), dtype=float)

        # Log-odds increments: weak evidence for free space, stronger for obstacle.
        free_update = -6 # aqui que eu regulo a velocidade da luzinha que sai no meu robo 
        occ_update = 3
        map_min = -20.0
        map_max = 20.0

        x_hit = x_r + distances * np.cos(theta_r + angles)
        y_hit = y_r + distances * np.sin(theta_r + angles)

     
        self.grid.add_map_points(x_hit, y_hit, occ_update)

        # If available, use lidar max range to avoid adding false obstacles at range limit.
        lidar_max_range = getattr(lidar, "max_range", np.inf)

        for distance, alpha in zip(distances, angles):
            if not np.isfinite(distance) or distance <= 0:
                continue

            ray_angle = theta_r + alpha

            # Free cells along the beam (stop one cell before measured obstacle when possible).
            free_distance = max(0.0, distance - (self.grid.resolution *8))
            x_free = x_r + free_distance * np.cos(ray_angle)
            y_free = y_r + free_distance * np.sin(ray_angle)
            self.grid.add_value_along_line(x_r, y_r, x_free, y_free, free_update) #

        # Saturation avoids numerical divergence after many updates.
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, map_min, map_max)

#     # def compute(self):
#     #     """ Useless function, just for the exercise on using the profiler """
#     #     # Remove after TP1

#     #     ranges = np.random.rand(3600)
#     #     ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

#     #     # Poor implementation of polar to cartesian conversion
#     #     points = []
#     #     for i in range(3600):
#     #         pt_x = ranges[i] * np.cos(ray_angles[i])
#     #         pt_y = ranges[i] * np.sin(ray_angles[i])
#     #         points.append([pt_x, pt_y])


