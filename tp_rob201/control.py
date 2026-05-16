""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values() #distances mesurées par le lidar
    laser_angles = lidar.get_ray_angles() #angles correspondants à ces distances, dans le référentiel du robot (0 devant, positif à gauche)

    angle_front = np.pi #couvre 360 degrés devant le robot (180° à gauche, 180° à droite)

    indices_frontais = np.where((laser_angles > -angle_front) & 
                                (laser_angles < angle_front)) #champ de vision
    
    dist_filt = laser_dist[indices_frontais] #filtrage de ces 'distances frontales' pour ne garder que celles qui sont dans le champ de vision

    if np.min(dist_filt) < 10: 
        #si un obstacle est trop proche, tourner aléatoirement à gauche ou à droite
        rotation_speed = random.uniform(-0.5, 0.5)
        speed = 0.0
    else: 
        #sinon, avancer tout droit
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

    # --- Parâmetros de Ajuste ---
    k_goal = 0.8         # Ganho de atração
    k_obs = 50000.0          # Ganho de repulsão (fórmula d^3 é forte, comece baixo)
    d_safe = 70.0         # Raio de influência dos obstáculos
    d_trans = 5.0        # Distância de transição (Linear -> Quadrática)
    tolerance = 0.2      # Condição de parada (m)
    
    v_max = 0.5          # Limite de velocidade linear
    w_max = 0.5         # Limite de velocidade angular

    x, y, theta = current_pose 
    x_goal, y_goal, _ = goal_pose
    q = np.array([x, y]) 
    q_goal = np.array([x_goal, y_goal]) #representa la posição du robot et du but sous forme de vecteurs 2D

    # --- 1. CÁLCULO DA DISTÂNCIA ---
    delta_goal = q_goal - q
    dist_goal = np.linalg.norm(delta_goal)
    
    # Condição de parada (item 2.1 do enunciado)
    if dist_goal < tolerance:
        return {"forward": 0.0, "rotation": 0.0}

    # --- 2. GRADIENTE ATRATIVO (∇f_att) ---
    # Implementação da transição Linear para Quadrática (suavização)
    if dist_goal > d_trans:
        # Forma Linear: Norma constante
        grad_att = (k_goal / dist_goal) * delta_goal
    else:
        # Forma Quadrática: Norma diminui proporcionalmente (Continuidade)
        grad_att = (k_goal / d_trans) * delta_goal

    # --- 2. GRADIENTE REPULSIVO ---
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    grad_rep_total = np.array([0.0, 0.0])

    # Filtramos todos os pontos que estão dentro da distância de segurança
    mask = (distances > 0) & (distances < d_safe)
    dist_v = distances[mask]
    angle_v = angles[mask]

    # Para cada ponto detectado, calculamos a repulsão e somamos
    for d_obs, a_obs in zip(dist_v, angle_v): #d_obs é a distância do obstáculo, a_obs é o ângulo do obstáculo no referencial do robô e o zip itera sobre ambos ao mesmo tempo
        # Direção do ponto no referencial GLOBAL
        world_alpha = theta + a_obs
        # Vetor unitário que aponta do robô para o ponto do obstáculo
        unit_vec = np.array([np.cos(world_alpha), np.sin(world_alpha)])
        
        # Fórmula do gradiente repulsivo 
        # (k_obs / d^3) * (1/d - 1/dsafe) * vetor_direção
        repulsion_mag = (k_obs / (d_obs**3)) * (1.0/d_obs - 1.0/d_safe)
        grad_rep_total += repulsion_mag * unit_vec

    # --- 4. COMPOSIÇÃO E COMANDO ---
    # Gradiente Total (Atração - Repulsão)
    grad_total = grad_att - grad_rep_total
    
    # Cálculo do erro de ângulo em relação ao vetor resultante
    target_angle = np.arctan2(grad_total[1], grad_total[0])
    angle_error = (target_angle - theta + np.pi) % (2 * np.pi) - np.pi

    # Velocidade Linear: Proporcional à norma do gradiente e ao alinhamento
    # np.cos(angle_error) garante que ele não corra se estiver olhando para o lado errado
    # np.tanh(dist_goal) ajuda a frear suavemente no alvo
    v_forward = v_max * np.cos(angle_error) * np.tanh(np.linalg.norm(grad_total))
    
    # Velocidade Angular: Proporcional ao erro de ângulo
    v_rotation = w_max * (angle_error / np.pi)

   
    

    # Retorno como Floats normalizados
    return {
        "forward": float(np.clip(v_forward, -0.5, 0.5)),
        "rotation": float(np.clip(v_rotation, -0.5, 0.5))
    }


