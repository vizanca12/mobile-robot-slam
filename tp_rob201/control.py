""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values() # Distances measured by the lidar
    laser_angles = lidar.get_ray_angles() # Angles corresponding to those distances, in the robot frame (0 ahead, positive to the left)

    angle_front = np.pi # Covers 360 degrees around the robot (180° to the left, 180° to the right)

    indices_frontais = np.where((laser_angles > -angle_front) & 
                                (laser_angles < angle_front)) # Field of view
    
    dist_filt = laser_dist[indices_frontais] # Filter these frontal distances to keep only those in the field of view

    if np.min(dist_filt) < 10: 
        # If an obstacle is too close, turn randomly left or right
        rotation_speed = random.uniform(-0.5, 0.5)
        speed = 0.0
    else: 
        # Otherwise, move straight ahead
        rotation_speed = 0.0
        speed = 0.5

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """

    # --- Tuning Parameters ---
    k_goal = 0.8         # Attraction gain
    k_obs = 50000.0      # Repulsion gain (the d^3 formula is strong, start low)
    d_safe = 70.0        # Obstacle influence radius
    d_trans = 5.0        # Transition distance (linear -> quadratic)
    tolerance = 0.2      # Stop condition (m)
    
    v_max = 0.5          # Linear speed limit
    w_max = 0.5          # Angular speed limit

    x, y, theta = current_pose 
    x_goal, y_goal, _ = goal_pose
    q = np.array([x, y]) 
    q_goal = np.array([x_goal, y_goal]) # Represents the robot and goal positions as 2D vectors

    # --- 1. DISTANCE COMPUTATION ---
    delta_goal = q_goal - q
    dist_goal = np.linalg.norm(delta_goal)
    
    # Stop condition (item 2.1 of the assignment)
    if dist_goal < tolerance:
        return {"forward": 0.0, "rotation": 0.0}

    # --- 2. ATTRACTIVE GRADIENT (∇f_att) ---
    # Linear-to-quadratic transition implementation (smoothing)
    if dist_goal > d_trans:
        # Linear form: constant norm
        grad_att = (k_goal / dist_goal) * delta_goal
    else:
        # Quadratic form: norm decreases proportionally (continuity)
        grad_att = (k_goal / d_trans) * delta_goal

    # --- 2. REPULSIVE GRADIENT ---
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    grad_rep_total = np.array([0.0, 0.0])

    # Filter all points within the safety distance
    mask = (distances > 0) & (distances < d_safe)
    dist_v = distances[mask]
    angle_v = angles[mask]

    # For each detected point, compute the repulsion and accumulate it
    for d_obs, a_obs in zip(dist_v, angle_v): # d_obs is the obstacle distance, a_obs is the obstacle angle in the robot frame, and zip iterates over both at the same time
        # Direction of the point in the GLOBAL frame
        world_alpha = theta + a_obs
        # Unit vector pointing from the robot to the obstacle point
        unit_vec = np.array([np.cos(world_alpha), np.sin(world_alpha)])
        
        # Repulsive gradient formula
        # (k_obs / d^3) * (1/d - 1/dsafe) * vetor_direção
        repulsion_mag = (k_obs / (d_obs**3)) * (1.0/d_obs - 1.0/d_safe)
        grad_rep_total += repulsion_mag * unit_vec

    # --- 4. COMPOSITION AND COMMAND ---
    # Total gradient (attraction - repulsion)
    grad_total = grad_att - grad_rep_total
    
    # Compute the angular error relative to the resulting vector
    target_angle = np.arctan2(grad_total[1], grad_total[0])
    angle_error = (target_angle - theta + np.pi) % (2 * np.pi) - np.pi

    # Linear velocity: proportional to the gradient norm and alignment
    # np.cos(angle_error) ensures it does not move forward if it is facing the wrong way
    # np.tanh(dist_goal) helps slow down smoothly near the target
    v_forward = v_max * np.cos(angle_error) * np.tanh(np.linalg.norm(grad_total))
    
    # Angular velocity: proportional to the angular error
    v_rotation = w_max * (angle_error / np.pi)

   
    

    # Return normalized floats
    return {
        "forward": float(np.clip(v_forward, -0.5, 0.5)),
        "rotation": float(np.clip(v_rotation, -0.5, 0.5))
    }


