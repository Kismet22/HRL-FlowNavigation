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
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
from termcolor import colored
from math import *
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import Arc, Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Port Check
def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None,
             target_center=None, target_position=None, start_center=None, 
             start_position=None, max_step=None, _include_flow=False, _plot_flow=False, _proccess_flow=False, _plot_task = "navigation",
             _random_range = 56, _flow_range=16, _switch_range = 12, _init_flow_num=0, _pos_normalize=True, _is_test=False, _is_random = True, _set_random=0,
             _state_dim=18, u_range=20.0, v_range=20.0, w_range=5.0, _forward_only = False, _obstacle_avoid=False, _is_switch=False, _theta_mode=False, u_clip=20.0, v_clip=20.0, w_clip=5.0, 
             _test_info=False, video_dir="./hrl_model_output/video_frames", _near_mode=False, _fov = 1.5 * np.pi, verbose=True, plot_mode="full", custom_trajectory=[], hover_points=[], _self_center=False):
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
        self.circles = [
            {"center": (53, 32), "radius": self.window_r/2},
            {"center": (53, 96), "radius": self.window_r/2},
            {"center": (109, 64), "radius": self.window_r/2}
        ]
        self.flow_num = _init_flow_num
        self._obstacle_avoid = _obstacle_avoid
        self._is_switch = _is_switch
        self._theta_mode = _theta_mode
        self._test_info = _test_info
        self._obstacle_mode = _near_mode
        self.end_state = "moving"
        self._fov = _fov
        self.flow_init = 0
        self.verbose = verbose
        self.plot_mode = plot_mode
        self.custom_trajectory = custom_trajectory
        self.hover_points = hover_points
        self._self_center = _self_center
        self._plot_task = _plot_task
        #################################################

        ###################################################
        # State Space
        self.include_flow = _include_flow
        self._pos_normalize = _pos_normalize
        # select mode: train or test
        self._is_test = _is_test 
        self._is_random = _is_random
        self._set_random = _set_random
        self.observation_dim = _state_dim
        self.state = np.zeros(self.observation_dim)
        self.pre_state = np.zeros(self.observation_dim)
        self.observation_space = Box(low=-1e6, high=1e6, shape=[self.observation_dim])
        #####################################################

        #####################################################
        # Action Space
        self.forward_mode = _forward_only
        if _forward_only:
            # self.action_dim = 2
            # low = np.array([0, 6], dtype=np.float32)
            # high = np.array([np.pi, 12], dtype=np.float32)
            self.action_dim = 1
            low = np.array([0], dtype=np.float32)
            high = np.array([np.pi], dtype=np.float32)
        else:
            if self._theta_mode:
                self.action_dim = 1
                low = np.array([-np.pi], dtype=np.float32)
                high = np.array([np.pi], dtype=np.float32)
            else:
                self.action_dim = 2
                low = np.array([-1.0, -1.0], dtype=np.float32)
                high = np.array([1.0, 1.0], dtype=np.float32)
        self.low = low
        self.high = high
        self.action_space = Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
        self.flow_force = []
        #####################################################

        #####################################################
        # Low-Level Controller
        self.lc_action_dim = 3
        lc_low = np.array([-float(u_range), -float(v_range), -float(w_range)], dtype=np.float32)
        lc_high = np.array([float(u_range), float(v_range), float(w_range)], dtype=np.float32)
        self.lc_low = lc_low
        self.lc_high = lc_high
        self.lc_action_space = Box(low=lc_low, high=lc_high, shape=(self.lc_action_dim,), dtype=np.float32)
        #####################################################


        #####################################################
        # Max Speed Limit Controller
        clip_low = np.array([-float(u_clip), -float(v_clip), -float(w_clip)], dtype=np.float32)
        clip_high = np.array([float(u_clip), float(v_clip), float(w_clip)], dtype=np.float32)
        self.clip_low = clip_low
        self.clip_high = clip_high
        #####################################################

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
        self.frame_id = 0
        self.video_dir = video_dir
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
        self.prev_agent_pos = np.array([0.0, 0.0])
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
        self.target_position_lc = self.target_position
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
        self._rolls = 0
        self._roll_count = 0
        self._collision_count = 0
        self._roll_punish = -50

        ############################
        # self.max_detect_dis = 24
        # self._old_dis_2_barricade = 24
        self.max_detect_dis = 24 - self.window_r/2
        self._old_dis_2_barricade = 24 - self.window_r/2
        self.safe_range_flag = False
        self.switch_range = _switch_range
        ############################

        self.old_direction = 0
        self.old_angle = 1
        #######################################################

        

        #########################################################
        # State Record
        # Agent Info
        self.agent_pos_history = []
        self.angle = 0
        self.vel_angle = 0
        self.speed = 0
        self.agent_velocity = []
        self.pressure = []
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
        # Low-Level Controller Info
        self.state_14 = []
        self.pre_state_14 = []
        #########################################################

        ###########################################################
        # Plot Settings
        self._plot_flow = _plot_flow
        self.frame_pause = 0.03

        # Create figure
        self.fig = plt.figure(figsize=(16, 9))

        # Create GridSpec: 1 row, 2 columns, left bigger than right
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.3)

        # Create axes
        self.ax_global = self.fig.add_subplot(gs[0])
        self.ax_local  = self.fig.add_subplot(gs[1])

        # backward compatibility
        self.ax = self.ax_global
        ###########################################################


        
        # ###########################################################
        # # Plot Settings
        # self._plot_flow = _plot_flow
        # self.frame_pause = 0.03
        # self.fig, self.ax = plt.subplots(figsize=(16, 9))
        # ###########################################################

        #############################################################
        # Port Settings
        flow_flag = 'true' if self.include_flow else 'false'
        # port = int(6000)
        while True:
            port = random.randint(6000, 8000)
            if not is_port_in_use(port):
                break

        port = port if network_port == None else network_port
        if local_port == None:
            command = (
                f'xvfb-run -a /home/zhengshizhan/workspace/processing-4.3/processing-java '
                # your file
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
        All returned values will be clipped to [-2.5, 2.5].

        Returns:
        v_x_grid, v_y_grid : Two numpy arrays where the small square region is filled with 0,
                            and values clipped to [-2.5, 2.5].
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

        # Clip values to [-2.5, 2.5]
        v_x_region = np.clip(v_x_region, -2.5, 2.5)
        v_y_region = np.clip(v_y_region, -2.5, 2.5)

        return v_x_region[:_cut_range, :_cut_range], v_y_region[:_cut_range, :_cut_range]


    def clip_action(self, action):
        # action = np.clip(action, self.lc_low, self.lc_high)
        action = np.clip(action, self.clip_low, self.clip_high)
        return action
    
    def sense(self,dist):
        sense_thresh = self.max_detect_dis+self.window_r/2
        if dist > sense_thresh:
            return 1.0
        return dist/ sense_thresh
    
    def set_lc_target(self, _lc_target):
        self.target_position_lc = _lc_target
    
    def set_target(self, _target):
        self.target_position = _target

    def step(self, action):
        self.safe_range_flag = False
        ########################
        # 1.1 Step Settings
        self.flow_force = []
        self.step_counter += 1
        action = self.clip_action(action)
        _truncated = False
        _terminated = False
        _collide = False
        _time_out = False
        _out_bound = False
        _reward = 0
        state_1 = self.state
        _dis_2_barricade = self._old_dis_2_barricade
        angle_norm = self.old_angle
        direction = self.old_direction
        self.action = action
        self.prev_agent_pos = self.agent_pos.copy()
        self.pre_state = self.state.copy()
        self.pre_state_14 = self.state_14.copy()
        self.end_state = "moving"
        ########################


        # ########################
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
            f_fluid = [state_1[17], state_1[18]]
            state_check = state_1.copy()
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}
        
        if np.isnan(state_1).any() or np.isinf(state_1).any():
            _out_bound = True
            _truncated = True
            _reward = -500
            self.end_state = "outbound"
            return self.state, _reward, _terminated, _truncated, {"error": "NaN in state"}
        
            
        # 2.3 Agent Info
        self.flow_force = f_fluid 
        self.agent_pos = [state_1[3], state_1[4]]
        self.angle = state_1[5]
        self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
        self.agent_velocity = [state_1[0], state_1[1]]
        self.vel_angle = abs(state_1[2]) # [1e-2, 1e-1]
        self.pressure = state_1[6:14]
        # ########################

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

        _fov = self._fov
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
            if self._self_center:
                vec_edge_local = R @ (edge_point_world - agent_pos)
            else:
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
            self.safe_range_flag = True
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

        ########################################################################################
        x_min, x_max = 4, self.x_range - 4
        y_min, y_max = 4, self.y_range - 4
        x, y = agent_pos
        dist_left = x - x_min
        dist_right = x_max - x
        dist_bottom = y - y_min
        dist_top = y_max - y

        state_border = np.array([
            self.sense(dist_left),
            self.sense(dist_right),
            self.sense(dist_bottom),
            self.sense(dist_top)
        ], dtype=np.float32)
        ########################################################################################

        ########################
        # Low-Level Controller State
        state_2[3], state_2[4] = self.target_position[0] - state_2[3], self.target_position[1] - state_2[4]
        self.state_14 = state_2
        ########################
        ########################
        # Relative Positon(Target & World Coordinate)
        if self._pos_normalize:
            near_threshold = self.max_detect_dis + self.window_r/2
            # === Extra fine-grained relative position ===
            if d < near_threshold:
                rel_x = (self.target_position[0] - state_1[3])
                rel_y = (self.target_position[1] - state_1[4])
                near_flag = 1.0
            else:
                rel_x = 0.0
                rel_y = 0.0
                near_flag = 0.0
            # === Normalized relative position for base state ===
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3])/near_threshold, (self.target_position[1] - state_1[4])/near_threshold
        else:
            # === Unnormalized relative position for base state ===
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3]), (self.target_position[1] - state_1[4])
        ########################
 
        # Step Reward
        #####################################################################################################
        if self.is_out_of_bounds():
            _out_bound = True
            _truncated = True
            if self._obstacle_mode:
                _reward = -100
            else:
                _reward = -500
            # _reward = 0
            self.end_state = "outbound"
            print(colored(f"**** Episode Finished At: {self.step_counter} **** Hit Boundary.", 'red'))
        

        # NaN or Inf state: outbound
        elif np.isnan(state_check).any() or np.isinf(state_check).any():
            _out_bound = True
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
                # if self._collision_count > 2:
                if self._collision_count > 0:
                    self.end_state = "collide"
                    _collide = True
                    _truncated = True
                    if self._obstacle_mode:
                        _reward = -50
                    else:
                        _reward = -500
                    print(colored(f"**** Episode Finished At: {self.step_counter} **** Too Many Collisions.", 'yellow'))
                else:
                    if self._obstacle_mode:
                        _reward = -50
                    else:
                        _reward = -250
                    print(colored(f"**** Forbidden Move **** Hit Circles{int(self._collision_count)}.", 'yellow'))
            ##################################################################################################### 

            #####################################################################################################
            elif self.step_counter >= self.t_step_max or self.done:
                self.end_state = "timelimit"
                _time_out = True
                _truncated = True
                if self._obstacle_mode:
                    _reward = -100
                else:
                    _reward = 0
                print(colored(f"**** Episode Finished D_2_T: {d}**** Reaches Env Time limit.", 'blue'))
            #####################################################################################################

            #####################################################################################################
            elif self.is_reach_target():
                self.end_state = "success"
                _terminated = True
                if self._obstacle_mode:
                    _reward = 200
                else:
                    _reward = 500
                if self.verbose:
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
                time_punish = -2 * self.dt
                # time_punish = -50 * self.dt
                # avoid_penalty = 25 * (_dis_2_barricade - self._old_dis_2_barricade)
                avoid_penalty = 50 * (_dis_2_barricade - self._old_dis_2_barricade)
                target_reward = 10 * (self.d_2_target - d)
                angle_reward = 5 * (self.old_angle - angle_norm) * abs(self.old_direction) * abs(direction)
                roll_swich_punish = -100 * abs(direction - self.old_direction) * abs(self.old_direction) * abs(direction)
                border_penalty = -50 * np.sum(1.0 - state_border)
                ########################

                ########################
                # mixed reward
                if self.observation_dim == 14 or self.observation_dim == 6 or self.observation_dim==11:
                    _reward = time_punish + target_reward
                if self.observation_dim == 16 or self.observation_dim == 8:
                    _reward = time_punish + target_reward + avoid_penalty
                if self.observation_dim == 18 or self.observation_dim == 10:
                    if self._obstacle_avoid:
                        if self._obstacle_mode:
                            _reward = time_punish + target_reward + avoid_penalty + angle_reward + roll_swich_punish
                        else:
                            # _reward = -50 * self.dt + target_reward + avoid_penalty + angle_reward + roll_swich_punish
                            _reward = time_punish + target_reward + avoid_penalty + angle_reward + roll_swich_punish
                    else:
                        _reward = time_punish + target_reward + border_penalty
                if self.observation_dim == 20:
                    _reward = time_punish + target_reward + avoid_penalty + border_penalty  
                if self.observation_dim == 22:
                    _reward = time_punish + target_reward + 2 * avoid_penalty + angle_reward + roll_swich_punish + border_penalty 
                if self.observation_dim == 25:
                    _reward = time_punish + target_reward + avoid_penalty + angle_reward + border_penalty + roll_swich_punish 
                    if d < near_threshold:
                        _reward = time_punish + 5 * target_reward
                ########################

                ########################
                # Roll Count & Terminate
                current_roll_count = self.angle / (2 * pi)
                full_turns = int(current_roll_count) - int(self._roll_count)

                if full_turns != 0:
                    self._rolls += 1
                    print(colored(f"**** Over Roll: {int(self._roll_count)} to {int(current_roll_count)} ****", 'blue'))
                    _reward += abs(full_turns) * self._roll_punish

                    # === too many rolls ===
                    max_roll_threshold = 5
                    if self._rolls > max_roll_threshold:
                        print(colored(f"**** Episode terminated: Over-Roll At: {self.step_counter} ****", 'red'))
                        _truncated = True
                        _reward = -500

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
            # no pressure + barricate sensor + roll direction guidance
            self.state = np.hstack([state_1[:6], barricade_state, direction, angle_norm]).astype(np.float32)
        elif self.observation_dim ==11:
            # no velocity
            self.state = np.hstack([state_1[3:6], state_1[6:14]]).astype(np.float32)
        elif self.observation_dim == 14:
            # only position
            self.state = state_1
        elif self.observation_dim == 16:
            # barricate sensor
            self.state = np.hstack([state_1, barricade_state]).astype(np.float32)
        elif self.observation_dim == 18:
            # roll direction guidance
            if self._obstacle_avoid:
                self.state = np.hstack([state_1, barricade_state, direction, angle_norm]).astype(np.float32)
            else:
                self.state = np.hstack([state_1, state_border]).astype(np.float32)
        elif self.observation_dim == 20:
            # edge sensor + barricate sensor
            self.state = np.hstack([state_1, barricade_state, state_border]).astype(np.float32)
        elif self.observation_dim == 22:
            # edge sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border]).astype(np.float32)
        elif self.observation_dim == 25:
            # near target sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border, rel_x, rel_y, near_flag]).astype(np.float32)
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
                if self._test_info:
                    return self.state, self.reward, _terminated, _out_bound, _time_out, _collide, _info, flow_input
                return self.state, self.reward, _terminated, _truncated, _info, flow_input
        if self._test_info:
            return self.state, self.reward, _terminated, _out_bound, _time_out, _collide, _info
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

    def reset(self, _flow_init=None, _start_position_init=None, _target_position_init=None, _verbose=True, _add_virtual_cylinders=False, _custom_cylinders=None, _hover_position=None):

        self.safe_range_flag = False
        self.frame_id = 0
        self.verbose = _verbose
        if self._plot_task == "hovering":
            self.hover_points = _hover_position
        ########################################################################################
        # Step1: Task Reset
        self.flow_force = []
        angle_norm = 1
        direction = 0
        self.step_counter = 0
        self.agent_pos_history = []
        self._roll_count = 0
        self._rolls = 0
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
            {"center": (centers[0], centers[1]), "radius": self.window_r / 2},
            {"center": (centers[2], centers[3]), "radius": self.window_r / 2},
            {"center": (centers[4], centers[5]), "radius": self.window_r / 2}
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



        # if _add_virtual_cylinders:
        #     x_mid = self.x_range/2

        #     virtual_circles = []
        #     for c in self.circles[:3]:
        #         x, y = c["center"]
        #         x_sym = 2 * x_mid - x
        #         virtual_circles.append({"center": (x_sym, y), "radius": c["radius"]})
        #     self.circles.extend(virtual_circles)
        #     print("✅ Added virtual symmetric cylinders (mirrored across x-axis center line).")

        # 1.3 Reset Agent & Target
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

            if self._is_switch and np.random.rand() < 0.5:
                print("Swapping start and target")
                tmp = self.start_position.copy()
                self.start_position = self.target_position.copy()
                self.target_position = tmp.copy()

            if self._obstacle_mode:
                print("Select start and target near obstacle")
                self.start_position, self.target_position = self.generate_start_and_target_pair()

        self.target_position_lc = self.target_position.copy()
        
        # # 1.3 Reset Agent & Target
        # ########################
        # if not self._is_fixed_target:
        #     print("Random Target")
        #     self.target_position = self.generate_point_outside_circles(self.target_default, self.random_range)
        #     self.target_position_lc = self.target_position
        # if not self._is_fixed_start:
        #     print("Random Start")
        #     self.start_position = self.generate_point_outside_circles(self.start_default, self.random_range)
        # ########################

        # # --- Switch Start and Target ---
        # if self._is_switch:
        #     if np.random.rand() < 0.5:
        #         print("Swapping start and target")
        #         tmp = self.start_position.copy()
        #         self.start_position = self.target_position.copy()
        #         self.target_position = tmp.copy()
        #         self.target_position_lc = self.target_position
        
        # if self._obstacle_mode:
        #     print("Select start and target Near Obstacle")
        #     # main_circle = random.choice(self.circles)["center"]
        #     # self.start_position, self.target_position = self.generate_start_and_target_pair(main_circle)
        #     self.start_position, self.target_position = self.generate_start_and_target_pair()
        
        # if _start_position_init is not None and isinstance(_start_position_init, np.ndarray):
        #     self.start_position = _start_position_init.copy()
        #     print("Start Position Reset Input")
        # if _target_position_init is not None and isinstance(_target_position_init, np.ndarray):
        #     self.target_position = _target_position_init.copy()
        #     print("Start Position Reset Input")

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
        # f_fluid = [state_1[17], state_1[18]]
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}

        # 2.3 Agent Info
        # self.flow_force = f_fluid 
        self.agent_pos = [state_1[3], state_1[4]]
        self.prev_agent_pos = [state_1[3], state_1[4]]
        self.angle = state_1[5]
        self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
        self.agent_velocity = [state_1[0], state_1[1]]
        self.pressure = state_1[6:14]
        self.vel_angle = abs(state_1[2]) # [1e-2, 1e-1]
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

        _fov = self._fov
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
            if self._self_center:
                vec_edge_local = R @ (edge_point_world - agent_pos)
            else:
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

        ########################################################################################
        x_min, x_max = 4, self.x_range - 4
        y_min, y_max = 4, self.y_range - 4
        x, y = agent_pos
        dist_left = x - x_min
        dist_right = x_max - x
        dist_bottom = y - y_min
        dist_top = y_max - y

        state_border = np.array([
            self.sense(dist_left),
            self.sense(dist_right),
            self.sense(dist_bottom),
            self.sense(dist_top)
        ], dtype=np.float32)
        ########################################################################################


        ########################
        # Low-Level Controller State
        state_2[3], state_2[4] = self.target_position[0] - state_2[3], self.target_position[1] - state_2[4]
        self.state_14 = state_2
        self.pre_state_14 = state_2
        ########################

        ########################
        # Relative Positon(Target & World Coordinate)
        if self._pos_normalize:
            near_threshold = self.max_detect_dis + self.window_r/2
            # === Extra fine-grained relative position ===
            if d < near_threshold:
                rel_x = (self.target_position[0] - state_1[3])
                rel_y = (self.target_position[1] - state_1[4])
                near_flag = 1.0
            else:
                rel_x = 0.0
                rel_y = 0.0
                near_flag = 0.0
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3])/near_threshold, (self.target_position[1] - state_1[4])/near_threshold
        else:
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3]), (self.target_position[1] - state_1[4])
        ########################

        ########################
        if self._pos_normalize:
            near_threshold = self.max_detect_dis + self.window_r/2
            # === Extra fine-grained relative position ===
            if d < near_threshold:
                rel_x = (self.target_position[0] - state_1[3])
                rel_y = (self.target_position[1] - state_1[4])
                near_flag = 1
            else:
                rel_x = 0.0
                rel_y = 0.0
                near_flag = 0
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
            # no pressure + barricate sensor + roll direction guidance
            self.state = np.hstack([state_1[:6], barricade_state, direction, angle_norm]).astype(np.float32)
        elif self.observation_dim ==11:
            # no velocity
            self.state = np.hstack([state_1[3:6], state_1[6:14]]).astype(np.float32)
        elif self.observation_dim == 14:
            # only position
            self.state = state_1
        elif self.observation_dim == 16:
            # barricate sensor
            self.state = np.hstack([state_1, barricade_state]).astype(np.float32)
        elif self.observation_dim == 18:
            # roll direction guidance
            if self._obstacle_avoid:
                self.state = np.hstack([state_1, barricade_state, direction, angle_norm]).astype(np.float32)
            else:
                self.state = np.hstack([state_1, state_border]).astype(np.float32)
        elif self.observation_dim == 20:
            # edge sensor + barricate sensor
            self.state = np.hstack([state_1, barricade_state, state_border]).astype(np.float32)
        elif self.observation_dim == 22:
            # edge sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border]).astype(np.float32)
        elif self.observation_dim == 25:
            # near target sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border, rel_x, rel_y, near_flag]).astype(np.float32)
        ########################
        self.pre_state = self.state.copy()
        
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            # Plot Flow
            if self._plot_flow:
                self.agent_pos_history.append(self.agent_pos)
                _save = True
                # self._render_frame()
                self._render_frame(_save=_save, save_start_target_map=True)
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

        state_ls = [
            state['vel_x'], state['vel_y'], state['vel_angle'],
            state['pos_x'], state['pos_y'], state['angle'],
            state['surfacePressures_1'], state['surfacePressures_2'], state['surfacePressures_3'],
            state['surfacePressures_4'], state['surfacePressures_5'], state['surfacePressures_6'],
            state['surfacePressures_7'], state['surfacePressures_8']
        ]

        state_ls += [
            state.get('f_ext_x', 0.0), state.get('f_ext_y', 0.0), state.get('m_ext', 0.0),
            state.get('f_fluid_x', 0.0), state.get('f_fluid_y', 0.0), state.get('m_fluid', 0.0),
            state.get('f_total_x', 0.0), state.get('f_total_y', 0.0), state.get('m_total', 0.0)
        ]

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

    
    def set_traj(self, traj_input):
        self.custom_trajectory = traj_input

    def set_hover(self, hov_input):
        self.hover_points = hov_input

    def close(self):
        if self.local_port == None:
            self.server.terminate()

    def _render_frame(self, _save=False, save_start_target_map=False):
        """
        Render a single frame of the environment.

        For navigation / hovering:
            - left: global full view
            - right: local agent-centered view

        For tracking:
            - single full global view

        Parameters
        ----------
        _save : bool
            Whether to save the current frame to disk.
        save_start_target_map : bool
            Whether to save a static start-target map (legend etc.)
        """
        if len(self.agent_pos_history) == 0:
            return

        # -------------------------------
        # Task flag
        plot_task = getattr(self, "_plot_task", "navigation")
        
        # -------------------------------
        # Tracking task: single global view
        if plot_task == "tracking":
            self.fig.clf()
            ax_global = self.fig.add_subplot(1, 1, 1)
            self.plot_env(ax_global, mode="full", local_view=False)
            plt.draw()
            plt.pause(self.frame_pause)

            if _save:
                os.makedirs(self.video_dir, exist_ok=True)
                save_path = os.path.join(self.video_dir, f"frame_{self.frame_id:05d}.png")
                self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
                self.frame_id += 1

            # Optional: save legend / static map
            if save_start_target_map:
                self.save_env_legend()
            return

        # -------------------------------
        # Navigation / Hovering: dual-view
        self.fig.clf()
        ax_global, ax_local = self.fig.subplots(
            1, 2,
            gridspec_kw={"width_ratios": [1.3, 1]}
        )

        # Global view (left)
        self.plot_env(
            ax_global,
            mode=self.plot_mode,   # original control logic
            local_view=False
        )

        # Local view (right)
        self.plot_env(
            ax_local,
            mode="simple",
            local_view=True,
            local_window=32
        )

        plt.tight_layout()
        plt.draw()
        plt.pause(self.frame_pause)

        # -------------------------------
        # Save frames
        if _save:
            os.makedirs(self.video_dir, exist_ok=True)

            # Combined dual-view frame (video)
            save_path = os.path.join(self.video_dir, f"frame_{self.frame_id:05d}.png")
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight")

            # Full global view
            fig_full, ax_full = plt.subplots(figsize=(16, 9))
            self.plot_env(ax_full, mode=self.plot_mode, local_view=False)
            full_path = os.path.join(self.video_dir, f"frame_{self.frame_id:05d}_full.png")
            fig_full.savefig(full_path, dpi=300, bbox_inches="tight")
            plt.close(fig_full)

            # Left global subview
            fig_left, ax_left = plt.subplots(figsize=(8, 9))
            self.plot_env(ax_left, mode=self.plot_mode, local_view=False)
            left_path = os.path.join(self.video_dir, f"frame_{self.frame_id:05d}_left.png")
            fig_left.savefig(left_path, dpi=300, bbox_inches="tight")
            plt.close(fig_left)

            # Right local subview
            fig_right, ax_right = plt.subplots(figsize=(6, 9))
            self.plot_env(ax_right, mode="simple", local_view=True, local_window=32)
            right_path = os.path.join(self.video_dir, f"frame_{self.frame_id:05d}_right.png")
            fig_right.savefig(right_path, dpi=300, bbox_inches="tight")
            plt.close(fig_right)

            # Increment frame counter
            self.frame_id += 1

        # -------------------------------
        # Optional: save static start-target map
        if save_start_target_map:
            self.save_env_legend()


    def plot_env(self, ax, mode="full", local_view=False, local_window=32):
        """
        Plot the environment.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on
        mode : str
            "full" or "simple"
        local_view : bool
            Whether to use agent-centered local zoom-in view
        local_window : float
            Window size (physical units) for local view
        """

        # -------------------------------
        # Task flag
        plot_task = getattr(self, "_plot_task", "navigation")
        assert plot_task in ("navigation", "hovering", "tracking"), \
            f"Unknown plot_task: {plot_task}"

        is_navigation = (plot_task == "navigation")
        is_hovering   = (plot_task == "hovering")
        is_tracking   = (plot_task == "tracking")

        # -------------------------------
        # Extract environment variables
        history = getattr(self, "agent_pos_history", [])
        start_point = history[0] if history else (0, 0)

        target = getattr(self, "target_position", None)
        current_target = getattr(self, "target_position_lc", None)

        circles = getattr(self, "circles", [])
        agent_pos = getattr(self, "agent_pos", (0, 0))
        agent_angle = getattr(self, "angle", 0)

        x_range = getattr(self, "x_range", 100)
        y_range = getattr(self, "y_range", 100)

        flow_x = getattr(self, "u_flow", np.zeros((y_range, x_range)))
        flow_y = getattr(self, "v_flow", np.zeros((y_range, x_range)))

        ax.clear()

        # -------------------------------
        # Compute vorticity
        dy, dx = y_range / flow_y.shape[1], x_range / flow_x.shape[0]
        dvdx = np.gradient(flow_y, dx, axis=0)
        dudy = np.gradient(flow_x, dy, axis=1)
        vorticity = dvdx - dudy

        vmin, vmax = (-2.0, 2.0) if local_view else (-1.0, 1.0)
        vorticity = np.clip(vorticity.T, vmin, vmax)

        cmap = plt.cm.seismic
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        x = np.linspace(0, x_range, flow_x.shape[0])
        y = np.linspace(0, y_range, flow_x.shape[1])
        X, Y = np.meshgrid(x, y)

        ax.pcolormesh(X, Y, vorticity, cmap=cmap, norm=norm, shading="auto", alpha=0.7, zorder=1)

        # -------------------------------
        # Add colorbar (always for left/global view)
        # if not local_view and not hasattr(ax, "_vorticity_colorbar_added"):
        #     from mpl_toolkits.axes_grid1 import make_axes_locatable
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="2%", pad=0.2)
        #     cbar = plt.colorbar(ax.collections[0], cax=cax)
        #     cbar.ax.tick_params(labelsize=10)
        #     ax._vorticity_colorbar_added = True
        

        # -------------------------------
        # Obstacles
        for circle in circles:
            ax.add_patch(
                plt.Circle(circle["center"], circle["radius"], color="black", alpha=0.3, zorder=3)
            )

        # -------------------------------
        # Trajectory
        if history:
            traj = history[-20:] if local_view else history
            hx, hy = zip(*traj)
            ax.plot(hx, hy, color="darkgreen", linewidth=3 if local_view else 2.5, zorder=4)

        # -------------------------------
        # Task-specific adjustments
        if is_navigation:
            # 原逻辑保持不变
            if mode == "full":
                ax.scatter(*start_point, color="darkgreen", s=50, zorder=5)
                if target is not None:
                    ax.scatter(*target, color="magenta", s=50, zorder=5)

                # Field of view
                radius = getattr(self, "max_detect_dis", 10)
                fov_angle = np.degrees(getattr(self, "_fov", np.pi / 4))
                angle_deg = np.degrees(agent_angle)
                center_angle = (angle_deg + 180) % 360
                start_angle = center_angle - fov_angle / 2
                end_angle = center_angle + fov_angle / 2
                wedge = Wedge(center=agent_pos, r=radius, theta1=start_angle, theta2=end_angle,
                            facecolor="darkgreen", alpha=0.2, zorder=2)
                ax.add_patch(wedge)

            if current_target is not None:
                size = 50 if not local_view else 80
                ax.scatter(*current_target, color="darkorange", s=size, zorder=6)

        elif is_hovering:
            # 左图（local_view=False）: 保留流场、障碍物、智能体、轨迹；新增hover_points
            if not local_view:
                hp = getattr(self, "hover_points", None)
                if hp is not None:
                    hp = np.array(hp).flatten()
                    if hp.size == 2:
                        ax.scatter(hp[0], hp[1], color="magenta", s=60, zorder=10)
            # 右图（local_view=True）: 绘制控制力和流场力
            else:
                if circles:
                    h = circles[0]["radius"]
                    w = h / 1.5
                    ax.add_patch(Ellipse(
                        xy=agent_pos, width=h, height=w, angle=np.degrees(agent_angle),
                        color="darkgreen", alpha=0.7, zorder=5
                    ))

                # Control & flow forces
                action = getattr(self, "action", None)
                flow_force = getattr(self, "flow_force", None)
                arrow_len_max = h * 1.2
                max_force_mag = np.sqrt(15**2 + 15**2)

                def scale_force(fx, fy):
                    mag = np.linalg.norm([fx, fy])
                    if mag < 1e-6:
                        return 0.0, 0.0
                    factor = arrow_len_max * mag / max_force_mag
                    return fx / mag * factor, fy / mag * factor

                if action is not None and len(action) >= 2:
                    fx_a, fy_a = scale_force(action[0], action[1])
                    if np.linalg.norm([fx_a, fy_a]) > 1e-6:
                        ax.arrow(agent_pos[0], agent_pos[1], fx_a, fy_a,
                                width=0.15, color="gold", zorder=8,
                                length_includes_head=True, head_width=1.2)

                if flow_force is not None and len(flow_force) >= 2:
                    fx_f, fy_f = scale_force(flow_force[0], flow_force[1])
                    if np.linalg.norm([fx_f, fy_f]) > 1e-6:
                        ax.arrow(agent_pos[0], agent_pos[1], fx_f, fy_f,
                                width=0.15, color="purple", zorder=8,
                                length_includes_head=True, head_width=1.2)

        # -------------------------------
        # Tracking task: 单图全局视图
        elif is_tracking:
            # 保留流场、障碍物、轨迹和智能体
            if circles:
                h = circles[0]["radius"]
                w = h / 1.5
                ax.add_patch(Ellipse(
                    xy=agent_pos, width=h, height=w, angle=np.degrees(agent_angle),
                    color="darkgreen", alpha=0.7, zorder=5
                ))
            # -------------------------------
            # 绘制参考轨迹（custom_trajectory）
            custom_traj = getattr(self, "custom_trajectory", None)
            if custom_traj is not None and len(custom_traj) > 0:
                # 确保是二维数组
                ct = np.array(custom_traj)
                if ct.shape[1] == 2:
                    # 绘制虚线轨迹
                    ax.plot(ct[:, 0], ct[:, 1], linestyle='--', color='orange', linewidth=2, zorder=3)

        # -------------------------------
        # Agent ellipse (navigation + left-hovering)
        if is_navigation or (is_hovering and not local_view):
            if circles:
                h = circles[0]["radius"]
                w = h / 1.5
                ax.add_patch(Ellipse(
                    xy=agent_pos, width=h, height=w, angle=np.degrees(agent_angle),
                    color="darkgreen", alpha=0.7, zorder=5
                ))

        # -------------------------------
        # View control
        ax.set_aspect("equal", adjustable="box")
        if local_view:
            cx, cy = agent_pos
            half = local_window / 2
            ax.set_xlim(max(cx - half, 0), min(cx + half, x_range))
            ax.set_ylim(max(cy - half, 0), min(cy + half, y_range))
        else:
            ax.set_xlim(0, x_range)
            ax.set_ylim(0, y_range)

        # -------------------------------
        # Hide axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    def save_env_legend(self):
        """
        Generate a standalone legend figure for the environment elements,
        without plotting the full environment.
        """

        # -------------------------------
        # Save path
        # -------------------------------
        os.makedirs(self.video_dir, exist_ok=True)
        save_path = os.path.join(self.video_dir, "legend.png")

        fig, ax = plt.subplots(figsize=(7, 2.2))

        # -------------------------------
        # Task flag
        # -------------------------------
        plot_task = getattr(self, "_plot_task", None)

        # ============================================================
        # Navigation task
        # ============================================================
        if plot_task == "navigation":
            ax.plot(
                [], [],
                color="darkgreen",
                linewidth=2.5,
                label="Trajectory"
            )
            ax.scatter([], [], s=40, color="darkgreen", label="Start")
            ax.scatter([], [], s=40, color="magenta", label="Target")
            ax.scatter([], [], s=40, color="darkorange", label="Sub-target")

            ax.legend(
                loc="center",
                ncol=4,
                frameon=False,
                fontsize=12,
                handlelength=2.5
            )

        # ============================================================
        # Hovering task
        # ============================================================
        elif plot_task == "hovering":
            ax.plot(
                [], [],
                color="darkgreen",
                linewidth=2.5,
                linestyle="-",
                label="Trajectory"
            )
            ax.scatter(
                [], [],
                s=40,
                color="magenta",
                label="Hover Point"
            )

            ax.legend(
                loc="center",
                ncol=2,
                frameon=False,
                fontsize=12,
                handlelength=2.5
            )

        # ============================================================
        # Tracking task
        # ============================================================
        elif plot_task == "tracking":
            ax.plot(
                [], [],
                color="darkgreen",
                linewidth=2.5,
                linestyle="-",
                label="Trajectory"
            )
            ax.plot(
                [], [],
                color="darkorange",
                linewidth=2.5,
                linestyle="--",
                label="Reference Trajectory"
            )

            ax.legend(
                loc="center",
                ncol=2,
                frameon=False,
                fontsize=12,
                handlelength=3.0
            )

        # ============================================================
        # Fallback
        # ============================================================
        else:
            print(f"Warning: Unknown plot_task '{plot_task}', legend skipped.")

        # -------------------------------
        # Finalize
        # -------------------------------
        ax.axis("off")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Legend saved to {save_path}")


    def save_start_target_map(self):
        """
        Save static map (flow field + start/target + obstacles) 
        and store flow data + environment metadata safely.
        """
        fig, ax = plt.subplots(figsize=(16, 9))

        x_range = self.x_range
        y_range = self.y_range
        flow_x = self.u_flow
        flow_y = self.v_flow

        # -------------------------------
        # Compute vorticity
        # -------------------------------
        dy, dx = y_range / flow_y.shape[1], x_range / flow_x.shape[0]
        dvdx = np.gradient(flow_y, dx, axis=0)
        dudy = np.gradient(flow_x, dy, axis=1)
        vorticity = dvdx - dudy

        vmin, vmax = -1.0, 1.0
        vorticity_clipped = np.clip(vorticity.T, vmin, vmax)
        cmap = plt.cm.seismic
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        x = np.linspace(0, x_range, flow_x.shape[0])
        y = np.linspace(0, y_range, flow_y.shape[1])
        X, Y = np.meshgrid(x, y)

        im = ax.pcolormesh(
            X, Y, vorticity_clipped,
            cmap=cmap, norm=norm,
            shading='auto', alpha=0.7, zorder=1
        )

        if not hasattr(ax, "_vorticity_colorbar_added"):
            cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02, extend='both')
            cbar.set_label("Vorticity", fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            ax._vorticity_colorbar_added = True

        # -------------------------------
        # Plot start/target regions
        # -------------------------------
        start_circle = plt.Circle(self.start_default, self.random_range,
                                color='darkgreen', alpha=0.3, zorder=5)
        target_circle = plt.Circle(self.target_default, self.random_range,
                                color='yellow', alpha=0.3, zorder=5)
        ax.add_patch(start_circle)
        ax.add_patch(target_circle)

        ax.text(*self.start_default, "Start Region",
                color='magenta', fontsize=15, ha='center', va='center', weight='bold')
        ax.text(*self.target_default, "Target Region",
                color='magenta', fontsize=15, ha='center', va='center', weight='bold')

        # Plot obstacles
        for circle in self.circles:
            ax.add_patch(plt.Circle(circle["center"], circle["radius"], color='red', alpha=0.3, zorder=4))

        # -------------------------------
        # Axes and limits
        # -------------------------------
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(0, self.y_range)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("X-axis", fontsize=12)
        ax.set_ylabel("Y-axis", fontsize=12)

        os.makedirs(self.video_dir, exist_ok=True)

        map_path = os.path.join(self.video_dir, "start_target_map.png")
        plt.savefig(map_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Start-Target map saved to {map_path}")

        flow_data_path = os.path.join(self.video_dir, "flow_field_data.npz")
        np.savez_compressed(
            flow_data_path,
            u_flow=self.u_flow,
            v_flow=self.v_flow,
            x_range=self.x_range,
            y_range=self.y_range,
            x_grid=x,
            y_grid=y
        )
        print(f"Flow field data saved to {flow_data_path}")

        def to_list(v):
            return v.tolist() if isinstance(v, np.ndarray) else v

        meta_data = {
            "start_default": to_list(self.start_default),
            "target_default": to_list(self.target_default),
            "random_range": float(self.random_range),
            "obstacles": [
                {
                    "center": to_list(circle["center"]),
                    "radius": float(circle["radius"])
                }
                for circle in self.circles
            ]
        }

        json_path = os.path.join(self.video_dir, "environment_meta.json")
        with open(json_path, "w") as f:
            json.dump(meta_data, f, indent=4)
        print(f"Environment metadata saved to {json_path}")


    def save_hover_task_setup(self):
        """
        Save a static map for the hover task setup:
        - Display target and 7 start points around it
        - Show local 24×24 flow field heatmap + quiver (no downsampling)
        - Each start point has unique color; all labels shown in right legend
        - Legend and colorbar placed separately, non-overlapping
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # -------------------------------
        # Flow field data
        # -------------------------------
        flow_x, flow_y = self.u_flow, self.v_flow
        x_range, y_range = self.x_range, self.y_range
        target = np.array(self.target_position)

        speed_field = np.sqrt(flow_x**2 + flow_y**2)
        x = np.linspace(0, x_range, flow_x.shape[0])
        y = np.linspace(0, y_range, flow_y.shape[1])
        X, Y = np.meshgrid(x, y)

        # -------------------------------
        # Crop around target (24×24 region)
        # -------------------------------
        half_window = 12
        x_min, x_max = target[0] - half_window, target[0] + half_window
        y_min, y_max = target[1] - half_window, target[1] + half_window

        x_mask = (X[0] >= x_min) & (X[0] <= x_max)
        y_mask = (Y[:, 0] >= y_min) & (Y[:, 0] <= y_max)
        X_crop = X[np.ix_(y_mask, x_mask)]
        Y_crop = Y[np.ix_(y_mask, x_mask)]
        Fx_crop = flow_x[np.ix_(x_mask, y_mask)].T
        Fy_crop = flow_y[np.ix_(x_mask, y_mask)].T
        speed_crop = speed_field[np.ix_(x_mask, y_mask)].T

        # -------------------------------
        # Background heatmap
        # -------------------------------
        vmin, vmax = 0.0, 2.5
        cmap = mcolors.LinearSegmentedColormap.from_list("red_white_blue", ["blue", "white", "red"])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(
            X_crop, Y_crop, np.clip(speed_crop, vmin, vmax),
            cmap=cmap, norm=norm, shading='auto', alpha=0.7, zorder=1
        )

        # -------------------------------
        # Full quiver arrows (no sampling)
        # -------------------------------
        mag = np.sqrt(Fx_crop**2 + Fy_crop**2)
        mag[mag == 0] = 1e-6
        ax.quiver(
            X_crop, Y_crop, Fx_crop / mag, Fy_crop / mag,
            color='black', scale=50, width=0.002, alpha=0.8, zorder=2
        )

        # -------------------------------
        # Target point
        # -------------------------------
        ax.scatter(*target, color='magenta', s=80, zorder=5)

        # -------------------------------
        # Start points (unique colors)
        # -------------------------------
        start_points = [(0, 0)]
        for r in [2, 4, 6]:
            for theta_deg in [0, 45, -45]:
                start_points.append((r, theta_deg))

        colors = plt.cm.tab10(np.linspace(0, 1, len(start_points)))
        handles = []
        for i, (r, theta) in enumerate(start_points):
            x_p = target[0] + r * np.cos(np.radians(theta))
            y_p = target[1] + r * np.sin(np.radians(theta))
            sc = ax.scatter(x_p, y_p, color=colors[i], s=60, zorder=4)
            handles.append(sc)

        # -------------------------------
        # Layout: leave room for legend
        # -------------------------------
        ax.set_position([0.08, 0.1, 0.68, 0.8])

        # -------------------------------
        # Colorbar
        # -------------------------------
        cbar_ax = fig.add_axes([0.78, 0.25, 0.02, 0.5])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label("Flow Speed (m/s)", fontsize=10)
        cbar.set_ticks(np.arange(vmin, vmax + 0.5, 0.5))
        cbar.ax.tick_params(labelsize=9)

        # -------------------------------
        # Legend (right of colorbar)
        # -------------------------------
        legend_labels = [f"r={r}, θ={theta}°" for r, theta in start_points]
        legend_ax = fig.add_axes([0.82, 0.25, 0.16, 0.5])
        legend_ax.axis("off")
        legend_ax.legend(
            handles, legend_labels,
            loc="upper right", frameon=False, fontsize=9
        )
        # legend_ax.set_title("Start Points", fontsize=10, fontweight="bold")

        # -------------------------------
        # Formatting
        # -------------------------------
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

        # -------------------------------
        # Save
        # -------------------------------
        os.makedirs(self.video_dir, exist_ok=True)
        save_path = os.path.join(self.video_dir, "hover_task_setup.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"Hover task setup map saved to {save_path}")


    def save_flow_field_map(self):
        """
        Save a simplified static map of the flow field:
        - Display flow field heatmap + quiver arrows
        - Include obstacles
        - No start or target regions
        """
        fig, ax = plt.subplots(figsize=(16, 9))

        # -------------------------------
        # Flow field (heatmap + quiver)
        # -------------------------------
        flow_x = self.u_flow
        flow_y = self.v_flow
        speed_field = np.sqrt(flow_x**2 + flow_y**2)

        x = np.linspace(0, self.x_range, flow_x.shape[0])
        y = np.linspace(0, self.y_range, flow_y.shape[1])
        X, Y = np.meshgrid(x, y)

        # Fixed color range
        vmin, vmax = 0.0, 2.5
        cmap = mcolors.LinearSegmentedColormap.from_list("red_white_blue", ["blue", "white", "red"])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Clip speed_field
        speed_clipped = np.clip(speed_field.T, vmin, vmax)

        # Heatmap
        im = ax.pcolormesh(
            X, Y, speed_clipped,
            cmap=cmap, norm=norm,
            shading='auto', alpha=0.6, zorder=1
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Flow Speed (m/s)", fontsize=12)
        tick_interval = 0.25
        ticks = np.arange(vmin, vmax + tick_interval, tick_interval)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=10)

        # -------------------------------
        # Quiver arrows (sampled for clarity)
        # -------------------------------
        desired_num_arrows_x, desired_num_arrows_y = 32, 12
        sample_rate_x = max(1, flow_x.shape[0] // desired_num_arrows_x)
        sample_rate_y = max(1, flow_y.shape[1] // desired_num_arrows_y)

        x_sample = x[::sample_rate_x]
        y_sample = y[::sample_rate_y]
        X_sample, Y_sample = np.meshgrid(x_sample, y_sample)

        sampled_flow_x = flow_x[::sample_rate_x, ::sample_rate_y]
        sampled_flow_y = flow_y[::sample_rate_x, ::sample_rate_y]
        magnitude = np.sqrt(sampled_flow_x**2 + sampled_flow_y**2)
        magnitude[magnitude == 0] = 1e-6

        ax.quiver(
            X_sample, Y_sample,
            sampled_flow_x / magnitude, sampled_flow_y / magnitude,
            color='black', scale=75, width=0.0015,
            headwidth=2.5, headlength=3, alpha=0.9,
            pivot='middle', zorder=3
        )

        # -------------------------------
        # Plot obstacles (if any)
        # -------------------------------
        if hasattr(self, "circles"):
            for circle in self.circles:
                ax.add_patch(plt.Circle(circle["center"], circle["radius"],
                                        color='red', alpha=0.3, zorder=4))

        # -------------------------------
        # Formatting
        # -------------------------------
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(0, self.y_range)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("X-axis", fontsize=12)
        ax.set_ylabel("Y-axis", fontsize=12)

        # -------------------------------
        # Save
        # -------------------------------
        os.makedirs(self.video_dir, exist_ok=True)
        save_path = os.path.join(self.video_dir, "flow_field_map.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Flow field map saved to {save_path}")
