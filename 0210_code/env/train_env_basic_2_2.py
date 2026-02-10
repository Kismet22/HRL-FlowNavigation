from xmlrpc.client import ServerProxy
import subprocess
import json
import time
import random
import numpy as np
import argparse
from gym.spaces import Box
import os
import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Wedge
from termcolor import colored
from math import *
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Obstacle sensing (in agent-centric coordinates)


# Port Check
def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None,
             target_center=None, target_position=None, start_center=None, start_position=None, max_step=None, 
             _include_flow=False, _plot_flow=False, _proccess_flow=False, 
             _random_range = 56, _switch_range = 24, _flow_range=16, _init_flow_num=0, _pos_normalize=True, 
             _is_test=False, _state_dim=18, _is_random = True, _is_switch = False, _set_random=0, plot_p = False, video_dir="./0809_rl_model_output/video_frames",
             u_range=20.0, v_range=20.0, w_range=5.0, u_clip=20.0, v_clip=20.0, w_clip=5.0, _fixed_test=False, _fov = np.pi):
        """
        Basic Info:
        - Expansion Factor: 16
        - Window Range: 320 * 128
        - Real Range: 20 * 8
        - Ellipse Agent: a = 1.5 * b; a = 4
        - Cylinder(3): r = 8
        """
        #################################################
        # Basic Info
        self.window_r = 1 * 16
        self.x_range = 20 * self.window_r
        self.y_range = 8 * self.window_r
        self.ellipse_a = self.window_r/4
        self.ellipse_b = self.ellipse_a / 1.5
        self.switch_range = _switch_range
        self.circles = [
            {"center": (53, 32), "radius": self.window_r/2},
            {"center": (53, 96), "radius": self.window_r/2},
            {"center": (109, 64), "radius": self.window_r/2}
        ]
        self.flow_num = _init_flow_num
        self.flow_init = 0
        self._is_test = _is_test
        self._is_random = _is_random
        self._set_random = _set_random
        self.plot_p = plot_p
        self.frame_id = 0
        self.video_dir = video_dir
        self._is_switch = _is_switch
        self._fixed_test = _fixed_test
        self.end_state = "moving"
        self.fov = _fov
        #################################################

        ###################################################
        # State Space
        self.include_flow = _include_flow
        self._pos_normalize = _pos_normalize
        self.observation_dim = _state_dim
        self.state = np.zeros(self.observation_dim)
        self.observation_space = Box(low=-1e6, high=1e6, shape=[self.observation_dim])

        # Action Space
        self.action_dim = 3

        ###############################
        low = np.array([-float(u_range), -float(v_range), -float(w_range)], dtype=np.float32)
        high = np.array([float(u_range), float(v_range), float(w_range)], dtype=np.float32)
        ###############################

        self.low = low
        self.high = high
        self.action_space = Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)

        #####################################################
        # Max Speed Limit
        clip_low = np.array([-float(u_clip), -float(v_clip), -float(w_clip)], dtype=np.float32)
        clip_high = np.array([float(u_clip), float(v_clip), float(w_clip)], dtype=np.float32)
        self.clip_low = clip_low
        self.clip_high = clip_high
        #####################################################

        # Sim Time
        self.step_counter = 0
        self.steps_default = 300
        self.dt = 0.075
        if max_step:
            self.t_step_max = max_step
        else:
            self.t_step_max = self.steps_default
        
        # Reward
        self.reward = 0

        # Done
        self.done = False
        ###################################################

        #####################################################
        # Port Settings
        self.action_interval = config.action_interval
        self.unwrapped = self
        self.unwrapped.spec = None
        self.local_port = local_port
        #####################################################
        

        #######################################################
        # Sim Settings
        self.agent_pos = np.array([0.0, 0.0])
        # Reach Range
        self.target_r = 0.5
        self.random_range = _random_range
        self.target_default = np.array([64, 64])
        # Target Range Center
        if target_center is not None:
            self.target_default = target_center
        self.start_default = np.array([220, 64])
        # Start Range Center
        if start_center is not None:
            self.start_default = start_center
        self.target_position = self.target_default
        self.start_position = self.start_default
        self._is_fixed_target = True
        self._is_fixed_start = True
        # Input Target
        if target_position is not None:
            self.target_default = target_position
            self.target_position = target_position
        else:
            # Random Select
            self._is_fixed_target = False
        # Input Start
        if start_position is not None:
            self.start_default = start_position
            self.start_position = start_position
        else:
            # Random Select
            self._is_fixed_start = False
        # Distance to Target
        self.d_2_target = 0
        self.d_2_target_min = np.inf
        self._roll_count = 0
        self._collision_count = 0
        self._roll_punish = -20

        ############################
        # self.max_detect_dis = 24
        # self._old_dis_2_barricade = 24
        self.max_detect_dis = 24 - self.window_r/2
        self._old_dis_2_barricade = 24 - self.window_r/2
        ############################

        self.old_direction = 0
        self.old_angle = 1
        #######################################################

        

        #########################################################
        # State Record
        # Agent Info
        self.agent_pos_history = []
        self.pressure_history = []
        self.angle = 0
        self.vel_angle = 0
        self.speed = 0
        self.pressure = []
        self.pre_pressure = []
        self.state_14 = []
        # Flow Info
        self.u_flow = []
        self.v_flow = []
        self._proccess_flow = _proccess_flow
        self.flow_speed = 0
        if self._proccess_flow:
            self.flow_range = _flow_range
            self.last_flow_x = np.zeros((2 * self.flow_range, 2 * self.flow_range))
            self.last_flow_y = np.zeros((2 * self.flow_range, 2 * self.flow_range))
        # Action Info
        self.action = []
        #########################################################

        
        ###########################################################
        # Plot Settings
        self._plot_flow = _plot_flow
        self.frame_pause = 0.03
        if self.plot_p:
            self.fig, (self.ax_env, self.ax_pressure) = plt.subplots(1, 2, figsize=(16, 9))
        else:
            self.fig, self.ax_env = plt.subplots(1, 1, figsize=(16, 9))
            self.ax_pressure = None
        ###########################################################

        #############################################################
        # Port Settings
        flow_flag = 'true' if self.include_flow else 'false'
        # port = int(6000)
        while True:
            port = random.randint(6000, 8000)
            if not is_port_in_use(port):
                break
            # port += 1
            
        port = port if network_port == None else network_port
        if local_port == None:
            command = (
                f'xvfb-run -a /home/zhengshizhan/workspace/processing-4.3/processing-java '
                f'--sketch=/home/zhengshizhan/project1/Navigation_understand-master/PinballCFD_server --run '
                f'{port} {flow_flag} {info}'
            )
            # Using subprocess to start
            self.server = subprocess.Popen(command, shell=True)
            # wait the server to start
            time.sleep(20)
            print("server start")
            self.proxy = ServerProxy(f"http://localhost:{port}/")
        else:
            self.proxy = ServerProxy(f"http://localhost:{local_port}/")
        
        if self._is_random:
            random.seed(None)
            np.random.seed(None)
        else:
            random.seed(self._set_random)
            np.random.seed(self._set_random)
        #############################################################


    def get_flow_velocity(self):
        """
        Compute the velocity field around the agent and store it in 16x16 grids.

        Parameters:
        r_large : Half-length of the larger square sensing range

        Returns:
        v_x_grid, v_y_grid : Two numpy arrays where the small square region is filled with 0.
        """
        v_x = self.u_flow
        v_y = self.v_flow
        x, y = self.agent_pos
        # Mask Range
        a = int(self.ellipse_a)
        r_large = int(self.flow_range)
        _cut_range = 2 * r_large
        # Flow Range
        H, W = v_x.shape

        # Checking NaN
        if np.isnan(x) or np.isnan(y):
            print(f"Warning: Agent position contains NaN (x={x}, y={y}).")
            return self.last_flow_x, self.last_flow_y

        # Checking Abnormal Position
        if x < -r_large or y < -r_large or x >= H + r_large or y >= W + r_large:
            print(f"Warning: Agent position out of bounds (x={x}, y={y}).")
            return self.last_flow_x, self.last_flow_y

        # Safety_margin
        safety_margin = 4
        pad_width = r_large + safety_margin

        # Mirror Padding
        v_x_padded = np.pad(v_x, pad_width=pad_width, mode='reflect')
        v_y_padded = np.pad(v_y, pad_width=pad_width, mode='reflect')
        x_padded = int(x + pad_width)
        y_padded = int(y + pad_width)

        v_x_region = v_x_padded[x_padded - r_large:x_padded + r_large + 1, 
                                y_padded - r_large:y_padded + r_large + 1]
        v_y_region = v_y_padded[x_padded - r_large:x_padded + r_large + 1, 
                                y_padded - r_large:y_padded + r_large + 1]
        mask = np.ones_like(v_x_region, dtype=bool)
        mask[r_large - a:r_large + a + 1, r_large - a:r_large + a + 1] = False
        v_x_region[~mask] = 0
        v_y_region[~mask] = 0
        return v_x_region[:_cut_range, :_cut_range], v_y_region[:_cut_range, :_cut_range]


    def clip_action(self, action):
        # action = np.clip(action, self.lc_low, self.lc_high)
        action = np.clip(action, self.clip_low, self.clip_high)
        return action
    
    
    def step(self, action):
        ########################
        # 1.1 Step Settings
        self.step_counter += 1
        action = self.clip_action(action)
        _truncated = False
        _terminated = False
        _reward = 0
        state_1 = self.state
        _dis_2_barricade = self._old_dis_2_barricade
        angle_norm = self.old_angle
        direction = self.old_direction
        self.action = action
        self.end_state = "moving"
        ########################

        ########################
        # 1.2 Simulator Act
        step_1 = float(action[0])
        step_2 = float(action[1])
        step_3 = float(action[2])
        action_json = {"v1": step_1, "v2": step_2, "v3": step_3}
        # action_json_0 = {"v1": 0, "v2": 0, "v3": 0}
        # Sim_Steps:10
        for i in range(self.action_interval):
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            if self.include_flow:
                step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)
                v_x = np.array(vflow_x, dtype=np.float32)
                v_y = np.array(vflow_y, dtype=np.float32)
            else:
                step_state = self.parseStep(res_str, self.include_flow)
            state_1 = np.array(step_state, dtype=np.float32)
            state_2 = np.array(step_state, dtype=np.float32)
            state_check = state_1.copy()
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}
        self.agent_pos = [state_1[3], state_1[4]]
        self.angle = state_1[5]
        self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
        self.vel_angle = abs(state_1[2]) # [1e-2, 1e-1]
        self.pressure = state_1[6:14]
        if self.plot_p:
            self.pressure_history.append(self.pressure.copy())
        ########################

        ########################################################################################
        # Distance to Target
        d = np.linalg.norm(self.agent_pos - self.target_position)
        ########################################################################################


        ########################################################################################
        # obstacle distance (filtered by ±90° within ship heading direction, local ship coords)

        agent_pos = np.array(self.agent_pos, dtype=np.float32)
        agent_angle = self.angle  # agent heading(rad);+:anticlockwise

        # Rotation matrix: rotate -agent_angle around Z-axis
        cos_a = np.cos(-agent_angle)
        sin_a = np.sin(-agent_angle)
        R = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])

        # Field of view ±90°
        _fov = self.fov

        _barricade_states = []

        # for circle in self.circles[:3]:
        for circle in self.circles:
            obstacle_pos = np.array(circle["center"], dtype=np.float32)
            obstacle_radius = circle.get("radius", 1.0)  # Use default radius if not provided

            # Obstacle vector in world coordinates (center to agent)
            vec_world = obstacle_pos - agent_pos
            center_distance = np.linalg.norm(vec_world)

            # Compute distance to edge of the obstacle
            edge_distance = center_distance - obstacle_radius
            if edge_distance < 1e-5 or edge_distance > self.max_detect_dis:
                continue

            # Unit vector pointing from center to agent
            unit_vec_world = vec_world / (center_distance + 1e-8)

            # Compute the edge point in world coordinates (closest point on the circle)
            edge_point_world = obstacle_pos - unit_vec_world * obstacle_radius

            # Transform edge point to agent's local coordinates
            # vec_edge_local = R @ (edge_point_world - agent_pos)
            vec_edge_local = (edge_point_world - agent_pos)

            # Check if within ±90° field of view
            forward_local = np.array([-1.0, 0.0], dtype=np.float32)
            unit_vec_local = vec_edge_local / (np.linalg.norm(vec_edge_local) + 1e-8)
            dot = np.dot(forward_local, unit_vec_local)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            if angle <= (_fov / 2):
                # Append: (distance to edge, local x, local y, world coordinates of edge)
                _barricade_states.append((
                    edge_distance,
                    vec_edge_local[0],
                    vec_edge_local[1],
                    edge_point_world
                ))

        # obstacle state(default)
        if len(_barricade_states) == 0:
            closest_distance = self.max_detect_dis
            if self._pos_normalize:
                closest_dx, closest_dy = -1.0, 1.0
            else:
                closest_dx, closest_dy = self.max_detect_dis, self.max_detect_dis
            direction = 0
            angle_rad = np.pi
        else:
            # sort by distance
            _barricade_states.sort(key=lambda x: x[0])
            closest_distance = _barricade_states[0][0]

            if self._pos_normalize:
                raw_dx = _barricade_states[0][1]
                raw_dy = _barricade_states[0][2]
                closest_dx = raw_dx / self.max_detect_dis
                closest_dx = np.clip(closest_dx, -1.0, 0.0)
                closest_dy = raw_dy / self.max_detect_dis
                closest_dy = np.clip(closest_dy, -1.0, 1.0)
            else:
                closest_dx = _barricade_states[0][1]
                closest_dy = _barricade_states[0][2]

            obstacle_pos = np.array(_barricade_states[0][3], dtype=np.float32)
            agent_pos = np.array(self.agent_pos, dtype=np.float32)
            target_pos = np.array(self.target_position, dtype=np.float32)

            vec1 = agent_pos - obstacle_pos
            vec2 = target_pos - obstacle_pos

            # normalize
            unit_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
            unit_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

            dot = np.dot(unit_vec1, unit_vec2)
            cross = np.cross(unit_vec1, unit_vec2)
            angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
            direction = np.sign(cross)

            # Stability filter for near-edge cases
            # angle_thresh = np.deg2rad(30)
            # if angle_rad < angle_thresh:
            #     direction = 0  
        _dis_2_barricade = min(closest_distance, self.max_detect_dis)
        barricade_state = np.array([closest_dx, closest_dy], dtype=np.float32)
        angle_norm = angle_rad / np.pi
        ########################################################################################

        ########################
        # Relative Positon(Target & World Coordinate)
        if self._pos_normalize:
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3])/(self.max_detect_dis+self.window_r/2), (self.target_position[1] - state_1[4])/(self.max_detect_dis+self.window_r/2)
        else:
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3]), (self.target_position[1] - state_1[4])
        ########################

        ########################
        # OG state
        state_2[3], state_2[4] = (self.target_position[0] - state_2[3]), (self.target_position[1] - state_2[4])
        self.state_14 = state_2
        ########################

        
        # Step Reward
        #####################################################################################################
        if self.is_out_of_bounds():
            _truncated = True
            _reward = -500
            self.end_state = "outbound"
            print(colored(f"**** Episode Finished At: {self.step_counter} **** Hit Boundary.", 'red'))
        
        # NaN or Inf state: outbound
        elif np.isnan(state_check).any() or np.isinf(state_check).any():
            _truncated = True
            _reward = -500
            self.end_state = "outbound"
            print(colored(f"**** Episode Finished At: {self.step_counter} **** Invalid state_1 detected (NaN/Inf).", 'red'))
        #####################################################################################################
        else:
            #####################################################################################################
            if self.is_in_circle(self.circles):
                self._collision_count += 1
                # end episode
                if self._collision_count > 0:
                    self.end_state = "collide"
                    _truncated = True
                    _reward = -500
                    print(colored(f"**** Episode Finished At: {self.step_counter} **** Too Many Collisions.", 'red'))
                else:
                    _reward = -250
                    print(colored(f"**** Forbidden Move **** Hit Circles{int(self._collision_count)}.", 'yellow'))
            ##################################################################################################### 

            #####################################################################################################
            elif self.step_counter >= self.t_step_max or self.done:
                _truncated = True
                _reward = 0
                self.end_state = "timelimit"
                print(colored(f"**** Episode Finished D_2_T: {d}**** Reaches Env Time limit.", 'blue'))
            #####################################################################################################

            #####################################################################################################
            elif self.is_reach_target():
                _terminated = True
                _reward = 500
                self.end_state = "success"
                print(colored(f"**** Episode Finished At: {self.step_counter} **** SUCCESS.", 'green'))
            #####################################################################################################

            #####################################################################################################
            else:
                ########################
                # _reward = -10 * self.dt - 10 * (d - self.d_2_target)
                ########################

                ########################
                if self.old_direction == 0 and direction !=0:
                    print(colored(f"**** Direction Confirm: {direction} ****", 'magenta'))   
                if self.old_direction != 0 and direction !=0 and self.old_direction != direction:
                    print(colored(f"**** Direction Switch: {self.old_direction} -> {direction} ****", 'magenta'))
                ########################

                ########################
                # 实验时,k=-50
                time_punish = -2 * self.dt
                avoid_penalty = 50 * (_dis_2_barricade - self._old_dis_2_barricade)
                target_reward = 10 * (self.d_2_target - d)
                angle_reward = 5 * (self.old_angle - angle_norm) * abs(self.old_direction) * abs(direction)
                roll_swich_punish = -100 * abs(direction - self.old_direction) * abs(self.old_direction) * abs(direction)
                ########################

                ########################
                # mixed reward
                if self.observation_dim == 14 or self.observation_dim == 6:
                    _reward = time_punish + target_reward
                if self.observation_dim == 18 or self.observation_dim == 10:
                    _reward = time_punish + target_reward + avoid_penalty + angle_reward + roll_swich_punish
                if self.observation_dim == 16 or self.observation_dim == 10:
                    _reward = time_punish + target_reward + avoid_penalty
                ########################

                ########################
                # Roll Count
                current_roll_count = self.angle / (2 * pi)
                full_turns = int(current_roll_count) - int(self._roll_count)
                if full_turns != 0:
                    print(colored(f"**** Over Roll:{int(self._roll_count)} to {int(current_roll_count)} ****", 'blue'))
                    _reward += abs(full_turns) * self._roll_punish
                self._roll_count = current_roll_count
                ########################
            #####################################################################################################
        
        # Update
        self.d_2_target = d
        self.d_2_target_min = min(self.d_2_target, self.d_2_target_min)
        self.reward = _reward
        self._old_dis_2_barricade = _dis_2_barricade
        self.old_angle = angle_norm
        self.old_direction = direction
        # self.pre_pressure = self.pressure

        ########################
        if self.observation_dim == 3:
            # only position
            self.state = state_1[3:6]  
        elif self.observation_dim == 6:
            # no pressure
            self.state = state_1[:6]
        elif self.observation_dim == 8:
            # no pressure + barricate sensor
            self.state = np.hstack([state_1[:6], barricade_state]).astype(np.float32)
        elif self.observation_dim == 10:
            # no pressure + obstacle
            self.state = np.hstack([state_1[:6], barricade_state, direction, angle_norm]).astype(np.float32)
        elif self.observation_dim ==11:
            # no velocity
            self.state = np.hstack([state_1[3:6], state_1[6:14]]).astype(np.float32)
        elif self.observation_dim == 14:
            # with pressure + no obstacle
            self.state = state_1
        elif self.observation_dim == 16:
            # with pressure + obstacle state
            self.state = np.hstack([state_1, barricade_state]).astype(np.float32)
        elif self.observation_dim == 18:
            # with pressure + roll direction guidance
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm]).astype(np.float32)
        ########################

        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            # Flow Plot
            if self._plot_flow:
                self.agent_pos_history.append(self.agent_pos)
                # _save = _terminated or _truncated
                _save = True
                self._render_frame(_save=_save)
            if self._proccess_flow:
                # Get Flow Speed
                v_x_grid, v_y_grid = self.get_flow_velocity()
                self.last_flow_x = v_x_grid
                self.last_flow_y = v_y_grid
                # Switch to PyTorch Tensor, Add batch dim
                flow_input = np.stack([v_x_grid, v_y_grid], axis=0) # Stack at batch_size dim
                return self.state, self.reward, _terminated, _truncated, _info, flow_input
        return self.state, self.reward, _terminated, _truncated, _info

    def _rand_in_half_circle(self, origin, r, _half_flag):
        d = sqrt(np.random.uniform(0, r ** 2))
        if _half_flag == 1:
            theta = np.random.uniform(0, pi)
        else:
            theta = np.random.uniform(pi, 2 * pi)
        return origin + np.array([d * cos(theta), d * sin(theta)])

    def _rand_in_circle(self, origin, r):
        d = sqrt(np.random.uniform(0, r ** 2))
        theta = np.random.uniform(0, 2 * pi)
        return origin + np.array([d * cos(theta), d * sin(theta)])

    def _is_in_circles(self, point):
        for circle in self.circles:
            center = np.array(circle["center"])

            #############################################
            radius = self.max_detect_dis + 2
            # radius = self.max_detect_dis + 4
            #############################################

            if np.linalg.norm(point - center) < radius:
                return True
        return False
    
    def half_generate_point_outside_circles(self, origin, r, _half_flag):
        while True:
            point = self._rand_in_half_circle(origin, r, _half_flag)
            if not self._is_in_circles(point):
                return point

    def generate_point_outside_circles(self, origin, r):
        while True:
            point = self._rand_in_circle(origin, r)
            if not self._is_in_circles(point):
                return point

    # def reset(self):
    def reset(self, _flow_init=None, _start_position_init=None, _target_position_init=None, _verbose=True, _add_virtual_cylinders=False, _custom_cylinders=None):
        ########################################################################################
        # Step1: Task Reset
        self.frame_id = 0
        angle_norm = 1
        direction = 0
        self.step_counter=0
        self.agent_pos_history = []
        self._roll_count = 0
        self._collision_count = 0
        self._old_dis_2_barricade = self.max_detect_dis
        _dis_2_barricade = self.max_detect_dis
        self.old_direction = 0
        self.old_angle = 1
        self.action = np.array([0, 0, 0], dtype=np.float32)
        self.end_state = "moving"
        self.flow_init = 0

        # 1.1 Reset Flow ID
        if not _flow_init:
            if self.flow_num:
                flow_init = self.flow_num
                print(f"Flow Reset Fixed:{flow_init}")
            else:
                flow_init = random.randint(1, 30)
                print(f"Flow Reset Random:{flow_init}")
        else:
            flow_init = _flow_init
            print(f"Flow Reset Input:{flow_init}")

        # 1.2 Reset Obstacle
        self.flow_init = flow_init
        obstacle_path = f'./PinballCFD_server/saved/init/init_{str(flow_init)}.txt'
        centers = np.loadtxt(obstacle_path)
        self.circles = [
            {"center": (centers[0], centers[1]), "radius": self.window_r/2},
            {"center": (centers[2], centers[3]), "radius": self.window_r/2},
            {"center": (centers[4], centers[5]), "radius": self.window_r/2}
        ]

        # ==== Add extra obstacles ====
        if _add_virtual_cylinders:
            use_custom = (
                _custom_cylinders is not None and
                isinstance(_custom_cylinders, (list, tuple)) and
                all(isinstance(p, (list, tuple)) and len(p) == 2 for p in _custom_cylinders)
            )

            if use_custom and len(_custom_cylinders) > 0:
                virtual_centers = list(_custom_cylinders)
                print(f"Added custom virtual cylinders at: {virtual_centers}")
            else:
                virtual_centers = [(200, 64), (200, 108), (200, 16)]
                print("Added default virtual cylinders at (200,64), (200,108), (200,16).")

            virtual_circles = [
                {"center": pos, "radius": self.window_r / 2} for pos in virtual_centers
            ]
            self.circles.extend(virtual_circles)
        
        # 1.3 Reset Agent & Target
        ########################
        if (_start_position_init is not None and _target_position_init is not None):
            self.start_position = np.array(_start_position_init, dtype=float)
            self.target_position = np.array(_target_position_init, dtype=float)
            print("Start Position Reset Input")
            print("Target Position Reset Input")
        else:
            if not self._is_fixed_target:
                print("Random Target")
                self.target_position = self.generate_point_outside_circles(self.target_default, self.random_range)
            if not self._is_fixed_start:
                print("Random Start")
                self.start_position = self.generate_point_outside_circles(self.start_default, self.random_range)
        ########################

        # --- Switch Start and Target ---
        if self._is_switch:
            if np.random.rand() < 0.5:
                print("Swapping start and target")
                tmp = self.start_position.copy()
                self.start_position = self.target_position.copy()
                self.target_position = tmp.copy()
                self.target_position_lc = self.target_position
        

        if self._fixed_test:
            print("Fixed Target")
            self.target_position = self.start_position.copy()

            
        print("target_pos:", self.target_position)
        print("start_pos:", self.start_position)
        agent_init_x = self.start_position[0]
        agent_init_y = self.start_position[1]
        ########################################################################################

        ########################################################################################
        # Step2:Reset true environment
        # action_json = {"v1": 0, "v2": 0, "v3": 0}
        # 2.1 Send Info To Sever
        action_json = {"v1": 0, "v2": 0, "v3": 0, "init_x": agent_init_x, "init_y": agent_init_y, "flow_num":float(flow_init)}

        # 2.2 Server Reset
        # print("[DEBUG] start_pos:", self.start_position)
        # print("[DEBUG] init_x:", agent_init_x, "type:", type(agent_init_x))
        # print("[DEBUG] init_y:", agent_init_y, "type:", type(agent_init_y))
        # print("[DEBUG] json_str:", json.dumps(action_json))

        res_str = self.proxy.connect.reset(json.dumps(action_json))
        if self.include_flow:
            # Unpacking
            step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)
            v_x = np.array(vflow_x, dtype=np.float32)
            v_y = np.array(vflow_y, dtype=np.float32)
        else:
            step_state = self.parseStep(res_str, self.include_flow)
        state_1 = np.array(step_state, dtype=np.float32)
        state_2 = np.array(step_state, dtype=np.float32)
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}

        # 2.3 Agent Info
        self.agent_pos = [state_1[3], state_1[4]]
        self.angle = state_1[5]
        self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
        self.pressure = state_1[6:14]
        if self.plot_p:
            self.pressure_history.append(np.zeros(8))
        ########################################################################################

        ########################################################################################
        # Distance to Target
        d = np.linalg.norm(self.agent_pos - self.target_position)
        ########################################################################################

        ########################################################################################
        # obstacle distance (filtered by ±90° within ship heading direction, local ship coords)

        agent_pos = np.array(self.agent_pos, dtype=np.float32)
        agent_angle = self.angle  # agent heading(rad);+:anticlockwise

        # Rotation matrix: rotate -agent_angle around Z-axis
        cos_a = np.cos(-agent_angle)
        sin_a = np.sin(-agent_angle)
        R = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])

        # Field of view ±90°
        _fov = self.fov

        _barricade_states = []

        # for circle in self.circles[:3]:
        for circle in self.circles:
            obstacle_pos = np.array(circle["center"], dtype=np.float32)
            obstacle_radius = circle.get("radius", 1.0)  # Use default radius if not provided

            # Obstacle vector in world coordinates (center to agent)
            vec_world = obstacle_pos - agent_pos
            center_distance = np.linalg.norm(vec_world)

            # Compute distance to edge of the obstacle
            edge_distance = center_distance - obstacle_radius
            if edge_distance < 1e-5 or edge_distance > self.max_detect_dis:
                continue

            # Unit vector pointing from center to agent
            unit_vec_world = vec_world / (center_distance + 1e-8)

            # Compute the edge point in world coordinates (closest point on the circle)
            edge_point_world = obstacle_pos - unit_vec_world * obstacle_radius

            # Transform edge point to agent's local coordinates
            # vec_edge_local = R @ (edge_point_world - agent_pos)
            vec_edge_local = (edge_point_world - agent_pos)

            # Check if within ±90° field of view
            forward_local = np.array([-1.0, 0.0], dtype=np.float32)
            unit_vec_local = vec_edge_local / (np.linalg.norm(vec_edge_local) + 1e-8)
            dot = np.dot(forward_local, unit_vec_local)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            if angle <= (_fov / 2):
                # Append: (distance to edge, local x, local y, world coordinates of edge)
                _barricade_states.append((
                    edge_distance,
                    vec_edge_local[0],
                    vec_edge_local[1],
                    edge_point_world
                ))

        # obstacle state(default)
        if len(_barricade_states) == 0:
            closest_distance = self.max_detect_dis
            if self._pos_normalize:
                closest_dx, closest_dy = -1.0, 1.0
            else:
                closest_dx, closest_dy = self.max_detect_dis, self.max_detect_dis
            direction = 0
            angle_rad = np.pi
        else:
            # sort by distance
            _barricade_states.sort(key=lambda x: x[0])
            closest_distance = _barricade_states[0][0]

            if self._pos_normalize:
                raw_dx = _barricade_states[0][1]
                raw_dy = _barricade_states[0][2]
                closest_dx = raw_dx / self.max_detect_dis
                closest_dx = np.clip(closest_dx, -1.0, 0.0)
                closest_dy = raw_dy / self.max_detect_dis
                closest_dy = np.clip(closest_dy, -1.0, 1.0)
            else:
                closest_dx = _barricade_states[0][1]
                closest_dy = _barricade_states[0][2]

            obstacle_pos = np.array(_barricade_states[0][3], dtype=np.float32)
            agent_pos = np.array(self.agent_pos, dtype=np.float32)
            target_pos = np.array(self.target_position, dtype=np.float32)

            vec1 = agent_pos - obstacle_pos
            vec2 = target_pos - obstacle_pos

            # normalize
            unit_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
            unit_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

            dot = np.dot(unit_vec1, unit_vec2)
            cross = np.cross(unit_vec1, unit_vec2)
            angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
            direction = np.sign(cross)

            # Stability filter for near-edge cases
            # 30 degrees
            # angle_thresh = np.deg2rad(30)
            # if angle_rad < angle_thresh:
            #     direction = 0  
        _dis_2_barricade = min(closest_distance, self.max_detect_dis)
        barricade_state = np.array([closest_dx, closest_dy], dtype=np.float32)
        angle_norm = angle_rad / np.pi
        ########################################################################################

        ########################
        # Relative Positon(Target & World Coordinate)
        if self._pos_normalize:
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3])/(self.max_detect_dis+self.window_r/2), (self.target_position[1] - state_1[4])/(self.max_detect_dis+self.window_r/2)
        else:
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3]), (self.target_position[1] - state_1[4])
        ########################

        ########################
        # OG state
        state_2[3], state_2[4] = (self.target_position[0] - state_2[3]), (self.target_position[1] - state_2[4])
        self.state_14 = state_2
        ########################

        self.d_2_target = d
        self.d_2_target_min = d
        self.old_angle = angle_norm
        self.old_direction = direction
        self._old_dis_2_barricade = _dis_2_barricade

        ########################
        if self.observation_dim == 3:
            # only position
            self.state = state_1[3:6]  
        elif self.observation_dim == 6:
            # no pressure
            self.state = state_1[:6]
        elif self.observation_dim == 8:
            # no pressure + barricate sensor
            self.state = np.hstack([state_1[:6], barricade_state]).astype(np.float32)
        elif self.observation_dim == 10:
            # no pressure + obstacle
            self.state = np.hstack([state_1[:6], barricade_state, direction, angle_norm]).astype(np.float32)
        elif self.observation_dim ==11:
            # no velocity
            self.state = np.hstack([state_1[3:6], state_1[6:14]]).astype(np.float32)
        elif self.observation_dim == 14:
            # with pressure + no obstacle
            self.state = state_1
        elif self.observation_dim == 16:
            # with pressure + obstacle state
            self.state = np.hstack([state_1, barricade_state]).astype(np.float32)
        elif self.observation_dim == 18:
            # with pressure + roll direction guidance
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm]).astype(np.float32)
        ########################

        
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            # Plot Flow
            if self._plot_flow:
                self.agent_pos_history.append(self.agent_pos)
                _save = True
                self._render_frame(_save = _save)
            if self._proccess_flow:
                # Get Flow Speed
                v_x_grid, v_y_grid = self.get_flow_velocity()
                self.last_flow_x = v_x_grid
                self.last_flow_y = v_y_grid
                # Switch to PyTorch Tensor, Add batch dim
                flow_input = np.stack([v_x_grid, v_y_grid], axis=0) # Stack at batch_size dim
                return self.state, _info, flow_input
        return self.state, _info
    

    def parseStep(self, info, _is_flow):
        all_info = json.loads(info)
        state = json.loads(all_info['state'][0])
        state_ls = [state['vel_x'], state['vel_y'], state['vel_angle'], state['pos_x'], state['pos_y'], state['angle'],
                    state['surfacePressures_1'], state['surfacePressures_2'], state['surfacePressures_3'],
                    state['surfacePressures_4'], state['surfacePressures_5'], state['surfacePressures_6'],
                    state['surfacePressures_7'], state['surfacePressures_8']]
        if _is_flow:
            flow_u = all_info.get('flow_u')
            flow_v = all_info.get('flow_v')
            return state_ls, flow_u, flow_v
        return state_ls

    def is_in_circle(self, circles):
        # Hitting obstacle
        for circle in circles:
            center = np.array(circle["center"])
            radius = circle["radius"] 
            a = self.ellipse_a
            b = self.ellipse_b
            distance = np.linalg.norm(self.agent_pos - center)
            if distance <= (a + radius):
                return True
        return False
    


    def is_out_of_bounds(self):
        _delta = 4
        x, y = self.agent_pos
        if not (_delta <= x <= self.x_range - _delta and _delta <= y <= self.y_range - _delta):
            print(f"Out of bounds: Agent_pos ({x:.2f}, {y:.2f})")
            return True
        return False

    
    def is_reach_target(self):
        x_c, y_c = self.agent_pos
        x_o, y_o = self.target_position
        # anticlockwise
        theta = self.angle
        a = self.ellipse_a
        b = self.ellipse_b
        # world coordinate to agent coordinate
        dx = x_o - x_c
        dy = y_o - y_c
        x_prime = dx * np.cos(theta) + dy * np.sin(theta)
        y_prime = -dx * np.sin(theta) + dy * np.cos(theta)
        # r = 12
        r = self.switch_range
        if self._is_test:
            # if (x_prime**2 / a**2 + y_prime**2 / b**2) <= 1:
            #     print(f"Success Steps: {self.step_counter}")
            # dx < a and dy < b
            return (x_prime**2 / a**2 + y_prime**2 / b**2) <= 1
        return dx**2 + dy**2 <= r**2

    def show_info(self, info):
        all_info = json.loads(info)
        state_json_str = all_info['state'][0]
        # state_json_str must be str or dic
        if isinstance(state_json_str, str):
            state_dict = json.loads(state_json_str)
        elif isinstance(state_json_str, dict):
            state_dict = state_json_str
        else:
            raise TypeError(f"Expected string or dict, got {type(state_json_str)}")
        # print labels
        for label in state_dict.keys():
            print(label)

    def terminate(self):
        pid = os.getpgid(self.server.pid)
        self.server.terminate()
        # Send the signal to all the process groups
        os.killpg(pid, signal.SIGTERM)  

    def close(self):
        if self.local_port == None:
            self.server.terminate()
    
    def _render_frame(self, _save=False):
        if len(self.agent_pos_history) == 0:
            return

        ax = self.ax_env
        ax.clear()
        self.plot_env(ax)
        plt.draw()
        plt.pause(self.frame_pause)

        if _save:
            os.makedirs(self.video_dir, exist_ok=True)
            save_path = os.path.join(self.video_dir, f"frame_{self.frame_id:05d}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Frame saved to {save_path}")
            self.frame_id += 1
    

    def plot_env(self, ax_env, ax_pressure=None, show_optional=True):
        """
        绘制端到端导航环境
        - 背景涡度分布
        - 障碍物、起点、目标
        - 智能体椭圆、轨迹
        - 可选显示 FOV、动作箭头、随机范围、网格、图例、标题
        - 可选压力曲线
        """

        # ==== 基础参数 ====
        history = self.agent_pos_history
        start_point = history[0]
        target = self.target_position
        circles = self.circles
        agent_pos = self.agent_pos
        agent_angle = self.angle
        x_range, y_range = self.x_range, self.y_range
        flow_x, flow_y = self.u_flow, self.v_flow

        ax = ax_env
        ax.clear()

        # ==== Vorticity (aligned with HRL) ====
        if flow_x is not None and flow_y is not None:
            # --- spacing (same as HRL) ---
            dy = y_range / flow_y.shape[1]
            dx = x_range / flow_x.shape[0]

            # --- vorticity definition ---
            dvdx = np.gradient(flow_y, dx, axis=0)
            dudy = np.gradient(flow_x, dy, axis=1)
            vorticity = dvdx - dudy

            # --- range (same default as HRL global view) ---
            vmin, vmax = -1.0, 1.0
            vorticity = np.clip(vorticity.T, vmin, vmax)

            # --- grid (strictly consistent) ---
            x = np.linspace(0, x_range, flow_x.shape[0])
            y = np.linspace(0, y_range, flow_x.shape[1])
            X, Y = np.meshgrid(x, y)

            # --- colormap & norm (same as HRL) ---
            cmap = plt.cm.seismic
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            ax.pcolormesh(
                X, Y, vorticity,
                cmap=cmap,
                norm=norm,
                shading="auto",
                alpha=0.7,
                zorder=1
            )


        # ==== 起点与目标 ====
        ax.scatter(*start_point, color='darkgreen', s=80, label='Start', zorder=5)
        ax.scatter(*target, color='magenta', s=80, label='Target', zorder=5)

        # ==== 障碍物 ====
        for circle in circles:
            ax.add_patch(Circle(circle["center"], circle["radius"], color='black', alpha=0.3, zorder=4))

        if show_optional:
            # ==== 感知扇形 ====
            agent_x, agent_y = agent_pos
            radius = getattr(self, 'max_detect_dis', 24)
            fov_angle = 180
            angle_deg = np.degrees(agent_angle)
            center_angle = (angle_deg + 180) % 360
            start_angle = (center_angle - fov_angle / 2) % 360
            end_angle = (center_angle + fov_angle / 2) % 360
            wedge = Wedge(center=(agent_x, agent_y), r=radius, theta1=start_angle, theta2=end_angle,
                        facecolor='darkgreen', alpha=0.2, zorder=3)
            ax.add_patch(wedge)

        # ==== 轨迹 ====
        if history:
            hx, hy = zip(*history)
            ax.plot(hx, hy, linestyle='-', color='darkgreen', linewidth=2.5, label='Path', zorder=4)

        # ==== 智能体椭圆 ====
        ellipse_h = circles[0]["radius"]
        ellipse_w = ellipse_h / 1.5
        ellipse = Ellipse(xy=agent_pos, width=ellipse_h, height=ellipse_w,
                        angle=np.degrees(agent_angle), color='darkgreen', alpha=0.7, zorder=5)
        ax.add_patch(ellipse)

        # if show_optional:
        #     # ==== 动作箭头 (可选) ====
        #     if hasattr(self, "action"):
        #         cos_a, sin_a = np.cos(agent_angle), np.sin(agent_angle)
        #         ax_local, ay_local = self.action[0], self.action[1]
        #         gx = ax_local * cos_a + ay_local * (-sin_a)
        #         gy = ax_local * sin_a + ay_local * cos_a
        #         ax.quiver(agent_pos[0], agent_pos[1], gx/2, gy/2,
        #                 angles='xy', scale_units='xy', scale=1,
        #                 width=0.002, headwidth=2.5, color='blue', zorder=6, label='Action')

        #     # ==== 坐标轴、网格、图例、标题 ====
        #     ax.set_aspect('equal', adjustable='box')
        #     ax.grid(True, linestyle='--', alpha=0.3)
        #     ax.legend(fontsize=9, loc='upper right', framealpha=0.8)
        #     ax.set_title(f"Navigation Trajectory – Step: {getattr(self, 'step_counter', 0)}", fontsize=12)

        ax.set_xlim(0, x_range)
        ax.set_ylim(0, y_range)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # ==== 压力曲线 ====
        if self.plot_p and ax_pressure is not None and hasattr(self, "pressure_history"):
            ax_pressure.clear()
            pressure_history = np.array(self.pressure_history)
            t = np.arange(len(pressure_history))
            for i in range(pressure_history.shape[1]):
                ax_pressure.plot(t, pressure_history[:, i], label=f"P{i}")
            ax_pressure.set_xlabel("Time Step", fontsize=11)
            ax_pressure.set_ylabel("Pressure", fontsize=11)
            ax_pressure.set_title("Pressure vs Time", fontsize=12)
            ax_pressure.legend(fontsize=8)
            ax_pressure.grid(True, linestyle='--', alpha=0.5)



    # def plot_env(self, ax_env, ax_pressure=None, sample_rate=10):
    #     """
    #     绘制环境（涡度场、智能体、目标、障碍物、轨迹等），可选绘制压力随时间变化。
    #     版本：绘制涡度分布，不绘制流场箭头。
    #     """

    #     # === 基础参数 ===
    #     history = self.agent_pos_history
    #     start_default = self.start_default
    #     target_default = self.target_default
    #     start_point = history[0]
    #     target = self.target_position
    #     circles = self.circles
    #     agent_pos = self.agent_pos
    #     agent_angle = self.angle
    #     x_range, y_range = self.x_range, self.y_range
    #     flow_x, flow_y = self.u_flow, self.v_flow
    #     random_range = self.random_range

    #     ax = ax_env
    #     ax.clear()

    #     # === 涡度分布（替代流场速度图） ===
    #     if flow_x is not None and flow_y is not None:
    #         # 计算涡度：dVy/dx - dVx/dy
    #         dy, dx = y_range / flow_y.shape[1], x_range / flow_x.shape[0]
    #         dvdx = np.gradient(flow_y, dx, axis=0)
    #         dudy = np.gradient(flow_x, dy, axis=1)
    #         vorticity = dvdx - dudy

    #         # 裁剪显示范围
    #         vmin, vmax = -1.0, 1.0
    #         vort_clipped = np.clip(vorticity.T, vmin, vmax)

    #         # 坐标网格
    #         x = np.linspace(0, x_range, flow_x.shape[0])
    #         y = np.linspace(0, y_range, flow_y.shape[1])
    #         X, Y = np.meshgrid(x, y)

    #         # 绘制涡度热图
    #         cmap = mcolors.LinearSegmentedColormap.from_list(
    #             "vort_blue_white_red", ["blue", "white", "red"]
    #         )
    #         norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    #         im = ax.pcolormesh(X, Y, vort_clipped, cmap=cmap, norm=norm,
    #                         shading='auto', alpha=0.8, zorder=1)

    #         # 添加颜色条（只添加一次）
    #         if not hasattr(ax, "_vorticity_colorbar_added"):
    #             divider = make_axes_locatable(ax)
    #             cax = divider.append_axes("right", size="2%", pad=0.2)
    #             cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    #             cbar.set_label("Vorticity", fontsize=11)
    #             tick_interval = 0.5
    #             ticks = np.arange(vmin, vmax + tick_interval, tick_interval)
    #             cbar.set_ticks(ticks)
    #             cbar.ax.tick_params(labelsize=9)
    #             ax._vorticity_colorbar_added = False

    #     # === 起点与目标点 ===
    #     # ax.scatter(*start_point, color='darkgreen', label='Start Position', zorder=5)
    #     ax.scatter(*target, color='magenta', label='Target Position', zorder=5)

    #     # === 障碍物 ===
    #     for circle in circles:
    #         ax.add_patch(plt.Circle(circle["center"], circle["radius"],
    #                                 color='black', alpha=0.3, zorder=4))

    #     # === 感知扇形（FOV） ===
    #     agent_x, agent_y = agent_pos
    #     radius = self.max_detect_dis
    #     fov_angle = 180
    #     angle_deg = np.degrees(agent_angle)
    #     center_angle = (angle_deg + 180) % 360
    #     start_angle = (center_angle - fov_angle / 2) % 360
    #     end_angle = (center_angle + fov_angle / 2) % 360
    #     wedge = patches.Wedge(
    #         center=(agent_x, agent_y),
    #         r=radius,
    #         theta1=start_angle,
    #         theta2=end_angle,
    #         facecolor='darkgreen',
    #         alpha=0.2,
    #         zorder=3
    #     )
    #     ax.add_patch(wedge)

    #     # === 轨迹 ===
    #     if history:
    #         hx, hy = zip(*history)
    #         ax.plot(hx, hy, linestyle='-', color='darkgreen', linewidth=2.5, label='Path')

    #     # === 随机范围区域 ===
    #     # ax.add_patch(plt.Circle(start_default, random_range, color='green', alpha=0.25, linestyle='--', fill=False))
    #     # ax.add_patch(plt.Circle(target_default, random_range, color='yellow', alpha=0.25, linestyle='--', fill=False))

    #     # === 智能体椭圆形体 ===
    #     ellipse_h = circles[0]["radius"]
    #     ellipse_w = ellipse_h / 1.5
    #     ellipse = Ellipse(
    #         xy=agent_pos, width=ellipse_h, height=ellipse_w,
    #         angle=np.degrees(agent_angle), color='darkgreen', alpha=0.7, zorder=5
    #     )
    #     ax.add_patch(ellipse)

    #     # === 智能体动作箭头 ===
    #     # cos_a, sin_a = np.cos(agent_angle), np.sin(agent_angle)
    #     # ax_local, ay_local = self.action[0], self.action[1]
    #     # gx = ax_local * cos_a + ay_local * (-sin_a)
    #     # gy = ax_local * sin_a + ay_local * cos_a
    #     # ax.quiver(agent_pos[0], agent_pos[1],
    #     #         gx / 2, gy / 2,
    #     #         angles='xy', scale_units='xy', scale=1,
    #     #         width=0.002, headwidth=2.5,
    #     #         color='blue', label='Action', zorder=6)

    #     # === 坐标轴与图例 ===
    #     ax.set_xlim(0, x_range)
    #     ax.set_ylim(0, y_range)
    #     # ax.set_aspect('equal', adjustable='box')
    #     # ax.grid(True, linestyle='--', alpha=0.5)
    #     # ax.legend(fontsize=9, loc='upper right', framealpha=0.8)
    #     # ax.set_title(f"Moving Trajectory – Step: {self.step_counter}", fontsize=12)

    #     # === 压力曲线 ===
    #     if self.plot_p and ax_pressure is not None:
    #         ax_pressure.clear()
    #         if hasattr(self, "pressure_history"):
    #             pressure_history = np.array(self.pressure_history)
    #             t = np.arange(len(pressure_history))
    #             for i in range(pressure_history.shape[1]):
    #                 ax_pressure.plot(t, pressure_history[:, i], label=f"P{i}")
    #             ax_pressure.set_xlabel("Time Step", fontsize=11)
    #             ax_pressure.set_ylabel("Pressure", fontsize=11)
    #             ax_pressure.set_title("Pressure vs Time", fontsize=12)
    #             ax_pressure.legend(fontsize=8)
    #             ax_pressure.grid(True, linestyle='--', alpha=0.5)
