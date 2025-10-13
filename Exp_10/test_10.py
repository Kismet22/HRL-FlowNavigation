import gym
import csv
import pandas as pd
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from termcolor import colored
import time
import torch.nn as nn
from math import *
from collections import deque


import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO
import hjbppo
from hjbppo import HJB_PPO
import icm_ppo
from icm_ppo import ICM_PPO


# Env
from env_new import train_env_upper_2
from env_new import train_env_upper_2_2
from env_new import train_env_upper_2_3
from env_new import train_env_basic_2_2

save_dir = './SuccessTest'
save_dir_1 = './model_output'
# plot_save_dir = './model_output/success_plot_withp.png'


####### DEVICE #######
device = my_ppo_net_1.device
# device = hjbppo.device
# device = icm_ppo.device

####### PREDICTOR #######
class Seq2SeqRelVelPredictor(nn.Module):
    def __init__(self, input_dim=9, hidden_size=128, lstm_layers=2, output_dim=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.output_dim = output_dim

        # Encoder: 处理 (pressure + orientation)
        self.encoder_input_fc = nn.Linear(input_dim, hidden_size)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # 直接输出相对速度 (vx, vy, vz)
        self.output_fc = nn.Linear(hidden_size, output_dim)

        self._initialize_weights()

    def forward(self, past_pressure, past_orient):
        """
        past_pressure: (B, T=10, 8)
        past_orient:   (B, T=10, 1)
        return: (B, 3)  # 当前相对速度
        """
        B, T, _ = past_pressure.shape

        # 拼接输入
        enc_input = torch.cat([past_pressure, past_orient], dim=-1)  # (B, 10, 9)
        enc_input = self.encoder_input_fc(enc_input)  # (B, 10, hidden_size)

        # 初始化隐藏状态
        h0 = torch.zeros(self.lstm_layers, B, self.hidden_size, device=enc_input.device)
        c0 = torch.zeros(self.lstm_layers, B, self.hidden_size, device=enc_input.device)

        # Encoder LSTM
        lstm_out, (h_n, c_n) = self.encoder_lstm(enc_input, (h0, c0))

        # 取最后时刻输出 (或者用 h_n[-1])
        last_out = lstm_out[:, -1, :]  # (B, hidden_size)

        # 映射到相对速度
        pred_rel_vel = self.output_fc(last_out)  # (B, 3)

        return pred_rel_vel

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

class SpeedPredictorWrapper:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = Seq2SeqRelVelPredictor(input_dim=9, hidden_size=256, lstm_layers=2, output_dim=3).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, past_pressure, past_pos_w):
        """
        past_pressure: np.array, shape (T, 8) 过去 T 步压力
        past_pos_w: np.array, shape (T, 1) 过去 T 步 z 角度
        return: np.array, shape (3,) 当前预测速度
        """
        # 转成 batch=1 tensor
        past_pressure_t = torch.tensor(past_pressure[None, :, :], dtype=torch.float32).to(self.device)
        past_pos_w = np.array(past_pos_w).reshape(-1, 1)  # 确保是 (T, 1)
        past_pos_w_t = torch.tensor(past_pos_w[None, :, :], dtype=torch.float32).to(self.device)
        pred_speed = self.model(past_pressure_t, past_pos_w_t)  # (1, 3)
        return pred_speed.cpu().numpy()[0]

####### HRL LOW #######
def compute_subgoal_forward(pos, angle, r):
    # print(f"target angle:{angle.item()}")
    x, y = pos[0], pos[1]
    angle_1 = angle.item() + np.pi/2
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def compute_subgoal(pos, angle, r):
    x, y = pos[0], pos[1]
    angle_1 = angle.item()
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def subgoal_reached(pos, target, threshold=4.0):
    dx = pos[0] - target[0]
    dy = pos[1] - target[1]
    return np.sqrt(dx**2 + dy**2) < threshold

def test():
    ########################### Mode 0: Single Test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    # action mode
    has_continuous_action_space = True
    # update epochs
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ
    gamma = 0.99  
    # learning rate for actor network
    lr_actor = 0.0001
    # learning rate for critic network  
    lr_critic = 0.0002
    # set std for action distribution when testing
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    ### Avoid Mode ###
    _is_avoid = True
    # _is_avoid = False

    ### Obstacle Mode ###
    # _is_near = True
    _is_near = False

    ### State Dim ###
    state_dim_mode = 10
    # lc_state_dim = 6
    lc_state_dim = 14

    ### Steps ###
    max_steps = 200
    max_ep_len = max_steps + 20
    max_ll_steps = 5
    #####################################################  

    
    ################ Other Settings ################
    ### set flow seed if required (0 : random flow; else : fixed flow) ###
    # flow_seed = 0
    flow_seed = 1

    ### set heading mode(True : forward only; else : all direction) ###
    # _is_head = True
    _is_head = False


    ### pre-trained high-level controller ###
    # ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
    # ppo_path_1 = f'./models/exp/lc_dim6.pth'

    # ppo_path = f'./models/exp/hrl/dim10/a20+fov270+near.pth'
    ppo_path = f'./models/exp/hrl/dim10_roll.pth'
    ppo_path_1 = f'./models/exp/lc_dim14_a20.pth'

    # ppo_path_1 = f'./models/exp/lc_dim{lc_state_dim}.pth'

    ### set random mode(True : real random mode; else : random.seed(random_seed)) ###
    true_random = True
    if true_random:
        random_seed = 0
    else:
        random_seed = 1
    
    test_mode = True
    switch_range = 8
    # test_mode = False
    # switch_range = 240

    # theta_mode = True
    theta_mode = False

    # switch_mode = True
    switch_mode = False

    _is_normalize = True
    # _is_normalize = False

    _fixed_test = True
    # _fixed_test = False

    _video_dir = "./Video_a20/video_frames"

    # input_max = 15.0
    # clip_max = 15.0

    input_max = 20.0
    clip_max = 20.0
    #####################################################


    # random point
    _start = np.array([float(220), float(64)])
    _target = np.array([float(64), float(64)])
    random_range = 32
    flow_seed = 0

    # env = train_env_upper_2_3.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _include_flow=True, _plot_flow=True, _proccess_flow=False,
    # _forward_only=_is_head, _obstacle_avoid=True, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max, _obstacle_mode=True)


    # env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _include_flow=True, _plot_flow=True, _proccess_flow=False,
    # _forward_only=_is_head, _obstacle_avoid=True, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max, _near_mode=True, _fov=1.5*np.pi)

    env = train_env_upper_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _include_flow=True, _plot_flow=True, _proccess_flow=False,
    _forward_only=_is_head, _obstacle_avoid=True, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max, _near_mode=True)

    # _start = np.array([float(240), float(64)])
    # _target = np.array([float(160), float(64)])
    # random_range = 32

    # env = train_env_upper_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    # _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    # _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=15.0, v_clip=15.0, _test_info=False, video_dir=_video_dir, _obstacle_mode=_is_near, _fixed_test=_fixed_test)


    # fixed point
    # _start = np.array([float(220), float(24)])
    # _target = np.array([float(160), float(32)])
    # random_range = 56
    # flow_seed = 1

    # _start = np.array([float(220.80355594), float(27.23576725)])
    # # _target = np.array([float(180.80355594), float(27.23576725)])
    # _target = np.array([float(220.80355594), float(27.23576725)])
    # random_range = 32

    # env = train_env_upper_2_2.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    # _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=speed_range, v_range=speed_range, 
    # _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip, _test_info=False, video_dir=_video_dir, _near_mode=_is_near, _fixed_test=_fixed_test)


    ################ Action Space Settings ################
    ### High-level Controller ###
    state_dim = state_dim_mode 
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    ### Low-level Controller ###
    lc_action_output_scale = np.array([])
    lc_action_output_bias = np.array([])
    if has_continuous_action_space:
        # low-level controller action space dimension
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    # initialize RL agent
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=200)
    # ppo_agent.load_full_icm(ppo_path)
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
            action_std_init=action_std_4_test, continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ######################################################

    ########################### Test ###########################
    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))

    subgoal_threshold = 1
    state, info = env.reset()

    # Store initial states
    state_14 = env.state_14
    agent_position = env.agent_pos
    prev_state = state.copy()
    prev_state_14 = state_14.copy()
    prev_agent_position = agent_position.copy()
    print("position_reset:", agent_position)
    target_position = env.target_position

    # Main episode loop
    t = 0
    hl_time_step = 0
    done = False

    while not done and t < max_ep_len:
        # === Get current high-level state ===
        high_state = state

        # === Decide whether to use fallback or normal high-level policy ===
        use_fallback = env.d_2_target < switch_range
        if use_fallback:
            # Compute angle from agent to final target
            vec_to_target = env.target_position - env.agent_pos
            target_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            print(colored(f"=== Fallback to target mode | d2target = {env.d_2_target:.2f} ===", "yellow"))
        else:
            high_action = ppo_agent.select_action(high_state)
            high_action = np.clip(high_action, env.low, env.high)
            hl_time_step += 1
            print(colored(f"=== High Level Step {hl_time_step}, Action: {high_action} ===", "blue"))

        # === Compute Subgoal ===
        agent_position = env.agent_pos
        if _is_head:
            angle_to_use = target_angle if use_fallback else high_action
            if use_fallback:
                r = np.linalg.norm(env.target_position - agent_position)
            else:
                r = 2
                # r = 5
            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=angle_to_use, r=r)
        else:
            if use_fallback:
                r = np.linalg.norm(env.target_position - agent_position)
                angle_to_use = target_angle
            else:
                r = 2
                # r = 5
                if not theta_mode:
                    vec = high_action
                    vec = vec / (np.linalg.norm(vec) + 1e-6)
                    angle_to_use = np.arctan2(vec[1], vec[0])
                else:
                    angle_to_use = high_action
            lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=angle_to_use, r=r)


        # === Heading Mode ===
        # agent_position = env.agent_pos
        # global_target = env.target_position
        # r = 2
        # dx, dy = global_target[0] - agent_position[0], global_target[1] - agent_position[1]
        # angle_2_target = np.arctan2(dy, dx)
        # lc_target_x = agent_position[0] + r * np.cos(angle_2_target)
        # lc_target_y = agent_position[1] + r * np.sin(angle_2_target)


        # === Normal Subgoal ===
        env.set_lc_target([lc_target_x, lc_target_y])
        print(f"=== Subgoal: ({lc_target_x}, {lc_target_y}) | Angle used: {angle_to_use} ===")

        # === Fixed Subgoal ===
        # env.set_lc_target(target_position)
        # [lc_target_x, lc_target_y] = target_position

        # === Low-level loop ===
        ll_step = 0
        done_lc = False
        while ll_step < max_ll_steps:
            agent_angle_old = env.angle

            # Low-level state update
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]

            if lc_state_dim == 3:
                low_state = low_state[3:6]
            elif lc_state_dim == 6:
                low_state = low_state[:6]
            elif lc_state_dim == 11:
                low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

            # Low-level action
            low_action = ppo_agent_lc.select_action(low_state)
            # low_action = [0.0, 0.0, 0.0]
            cliped_low_action = env.clip_action(low_action)
            print(f"=== Low Level Step {t}, Action: {low_action}")
            print(f"===               -> Clip_Action: {cliped_low_action} ===")

            # Step env
            state, reward, terminated, truncated, info = env.step(low_action)
            done = terminated or truncated
            # done = False
            agent_position = env.agent_pos
            state_14 = env.state_14

            # Check subgoal success
            dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
            if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                done_lc = True

            t += 1
            ll_step += 1

            if done_lc:
                ppo_agent_lc.buffer.clear()
                break

        ppo_agent.buffer.clear()

    env.close()
    print("Total_Steps:", t)
    ##############################################################################

def test_1():
    ########################### Mode 1: RL Test ###########################
    print("============================================================================================")

    # easy
    # ppo_path = './models/easy_task/0515_5.pth'
    # state_dim_mode = 18

    ppo_path = './models/exp/rl_dim6_a15_forward_easy.pth'
    state_dim_mode = 6
    max_steps = 400
    max_ep_len = max_steps + 20
    ### set flow seed if required (0 : random flow; else : fixed flow) ###
    # flow_seed = 0
    flow_seed = 1

    max_steps = 300
    max_ep_len = max_steps + 10
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    _start = np.array([float(160), float(32)])
    _target = np.array([float(160), float(32)])
    random_range = 56

    # _start = np.array([float(240), float(64)])
    # _target = np.array([float(160), float(64)])
    # random_range = 32

    ### set random mode(True : real random mode; else : random.seed(random_seed)) ###
    true_random = True
    if true_random:
        random_seed = 0
    else:
        random_seed = 1


    _is_normalize = True
    # _is_normalize = False

    test_mode = True
    switch_range = 1
    # test_mode = False
    # switch_range = 240

    env = train_env_basic_2_2.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=15.0, v_range=15.0, 
    _is_switch=False, u_clip=15.0, v_clip=15.0, video_dir="./Video_RL_Fixed/video_frames", _fixed_test=True)


    ########################### State&Action ###########################
    # state space dimension
    # continuous action space; else discrete
    has_continuous_action_space = True
    # state_dim = env.observation_space.shape[0]
    state_dim = state_dim_mode
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        # action scale
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        # discrete
        action_dim = env.action_space.n
    ######################################################

    ########################### PPO ###########################
    # initialize parameter
    # update policy for K epochs in one PPO update
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ  
    gamma = 0.99
    # learning rate for actor network  
    lr_actor = 0.0001  
    # learning rate for critic network
    lr_critic = 0.0002  
    # set std for action distribution when testing.
    action_std_4_test = 0.04  
    # initialize RL agent
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=200)
    # ppo_agent.load_full_icm(ppo_path)
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    # ppo_agent_1 = Classic_PPO(14, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias)
    # ppo_agent_1.load_full(ppo_path_1)
    ######################################################

    ########################### Test ###########################
    speed_ppo = []
    action_ppo = []
    pressure_ppo = []
    # 1 env reset
    state, info = env.reset()
    print("position_reset:", env.agent_pos)

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):
        # 2 env step
        action = ppo_agent.select_action(state)
        print(f"Step {i}, Action: {action}")
        action_ppo.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        # 3 save info
        ep_return_ppo += reward
        # if terminated or truncated:
        #     break
        # 4 clear buffer
        ppo_agent.buffer.clear()
    env.close()
    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)
    ######################################################


def test_2():
    ########################### Mode 2: Predictor Test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################

    ################ Predict Model ################
    _is_predict = False
    predictor = SpeedPredictorWrapper(model_path='./predictor_09/model_0924.pth')
    input_len = 10
    past_pressure_seq = deque(maxlen=input_len)
    past_pos_w_seq = deque(maxlen=input_len)
    #####################################################  

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    # action mode
    has_continuous_action_space = True
    # update epochs
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ
    gamma = 0.99  
    # learning rate for actor network
    lr_actor = 0.0001
    # learning rate for critic network  
    lr_critic = 0.0002
    # set std for action distribution when testing
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    ### Avoid Mode ###
    _is_avoid = True
    # _is_avoid = False

    ### Obstacle Mode ###
    # _is_near = True
    _is_near = False

    ### State Dim ###
    # state_dim_mode = 22
    # state_dim_mode = 18
    # state_dim_mode = 16
    # state_dim_mode = 14
    state_dim_mode = 6
    # lc_state_dim = 14
    lc_state_dim = 6

    ### Steps ###
    max_steps = 500
    max_ep_len = max_steps + 20
    max_ll_steps = 5
    #####################################################  

    
    ################ Other Settings ################
    ### set flow seed if required (0 : random flow; else : fixed flow) ###
    # flow_seed = 0
    flow_seed = 1

    ### set heading mode(True : forward only; else : all direction) ###
    # _is_head = True
    _is_head = False


    ### pre-trained high-level controller ###
    ppo_path = f'./models/exp/hrl/dim{state_dim_mode}+noobs+normalize.pth'
    ppo_path_1 = f'./models/exp/lc_dim{lc_state_dim}.pth'

    ### set random mode(True : real random mode; else : random.seed(random_seed)) ###
    true_random = True
    if true_random:
        random_seed = 0
    else:
        random_seed = 1
    
    test_mode = True
    switch_range = 1e-3
    # test_mode = False
    # switch_range = 240

    # theta_mode = True
    theta_mode = False

    # switch_mode = True
    switch_mode = False

    _is_normalize = True
    # _is_normalize = False

    _fixed_test = True
    # _fixed_test = False

    _video_dir = "./Video_Fixed/video_frames"

    speed_range = 15.0
    speed_clip = 15.0
    #####################################################


    # random point
    # _start = np.array([float(220), float(64)])
    # _target = np.array([float(64), float(64)])
    # random_range = 56
    # flow_seed = 0

    # _start = np.array([float(240), float(64)])
    # _target = np.array([float(160), float(64)])
    # random_range = 32

    # env = train_env_upper_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    # _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    # _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=15.0, v_clip=15.0, _test_info=False, video_dir=_video_dir, _obstacle_mode=_is_near, _fixed_test=_fixed_test)


    # fixed point
    _start = np.array([float(220), float(64)])
    _target = np.array([float(160), float(64)])
    random_range = 56
    flow_seed = 1

    # _start = np.array([float(220.80355594), float(27.23576725)])
    # # _target = np.array([float(180.80355594), float(27.23576725)])
    # _target = np.array([float(220.80355594), float(27.23576725)])
    # random_range = 32

    env = train_env_upper_2_2.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=speed_range, v_range=speed_range, 
    _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip, _test_info=False, video_dir=_video_dir, _near_mode=_is_near, _fixed_test=_fixed_test)


    ################ Action Space Settings ################
    ### High-level Controller ###
    state_dim = state_dim_mode 
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    ### Low-level Controller ###
    lc_action_output_scale = np.array([])
    lc_action_output_bias = np.array([])
    if has_continuous_action_space:
        # low-level controller action space dimension
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    # initialize RL agent
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=200)
    # ppo_agent.load_full_icm(ppo_path)
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
            action_std_init=action_std_4_test, continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ######################################################

    ########################### Test ###########################
    # Predictor Info
    past_pressure_seq.clear()
    past_pos_w_seq.clear()
    ##############################

    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))


    # Main episode loop
    t = 0
    hl_time_step = 0
    done = False
    subgoal_threshold = 1
    
    state, info = env.reset()
    agent_position = env.agent_pos

    # Initial observation
    ##############################
    for _ in range(input_len):
        past_pressure_seq.append(env.pressure.copy())
        past_pos_w_seq.append(env.angle.copy() )
    # predict speed
    pred_speed = predictor.predict(past_pressure=np.array(past_pressure_seq), past_pos_w=np.array(past_pos_w_seq))
    print("position_reset:", agent_position)
    print(colored(f"Predicted speed: {pred_speed}", "red"))
    print(colored(f"True speed: {env.speed}", "green"))
    if _is_predict and state_dim_mode != 3:
        state[:3] = pred_speed
    ###############################


    # Store initial states
    state_14 = env.state_14
    prev_state = state.copy()
    prev_state_14 = state_14.copy()
    prev_agent_position = agent_position.copy()
    target_position = env.target_position

    while not done and t < max_ep_len:
        # === Get current high-level state ===
        high_state = state
        # === Decide whether to use fallback or normal high-level policy ===
        use_fallback = env.d_2_target < switch_range
        if use_fallback:
            # Compute angle from agent to final target
            vec_to_target = env.target_position - env.agent_pos
            target_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            print(colored(f"=== Fallback to target mode | d2target = {env.d_2_target:.2f} ===", "yellow"))
        else:
            high_action = ppo_agent.select_action(high_state)
            high_action = np.clip(high_action, env.low, env.high)
            hl_time_step += 1
            print(colored(f"=== High Level Step {hl_time_step}, Action: {high_action} ===", "blue"))

        # === Compute Subgoal ===
        agent_position = env.agent_pos
        if _is_head:
            angle_to_use = target_angle if use_fallback else high_action
            if use_fallback:
                r = np.linalg.norm(env.target_position - agent_position)
            else:
                # r = 3
                r = 5
            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=angle_to_use, r=r)
        else:
            if use_fallback:
                r = np.linalg.norm(env.target_position - agent_position)
                angle_to_use = target_angle
            else:
                # r = 3
                r = 5
                if not theta_mode:
                    vec = high_action
                    vec = vec / (np.linalg.norm(vec) + 1e-6)
                    angle_to_use = np.arctan2(vec[1], vec[0])
                else:
                    angle_to_use = high_action
            lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=angle_to_use, r=r)


        # === Heading Mode ===
        # agent_position = env.agent_pos
        # global_target = env.target_position
        # r = 3
        # dx, dy = global_target[0] - agent_position[0], global_target[1] - agent_position[1]
        # angle_2_target = np.arctan2(dy, dx)
        # lc_target_x = agent_position[0] + r * np.cos(angle_2_target)
        # lc_target_y = agent_position[1] + r * np.sin(angle_2_target)


        # === Normal Subgoal ===
        env.set_lc_target([lc_target_x, lc_target_y])
        print(f"=== Subgoal: ({lc_target_x}, {lc_target_y}) | Angle used: {angle_to_use} ===")

        # === Fixed Subgoal ===
        # env.set_lc_target(target_position)
        # [lc_target_x, lc_target_y] = target_position

        # === Low-level loop ===
        ll_step = 0
        done_lc = False
        while ll_step < max_ll_steps:
            agent_angle_old = env.angle

            # Low-level state update
            low_state = state_14.copy()
            if _is_predict:
                low_state[:3] = pred_speed
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]

            if lc_state_dim == 3:
                low_state = low_state[3:6]
            elif lc_state_dim == 6:
                low_state = low_state[:6]
            elif lc_state_dim == 11:
                low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

            # Low-level action
            low_action = ppo_agent_lc.select_action(low_state)
            cliped_low_action = env.clip_action(low_action)
            print(f"=== Low Level Step {t}, Action: {low_action}")
            print(f"===               -> Clip_Action: {cliped_low_action} ===")

            # Step env
            state, reward, terminated, truncated, info = env.step(low_action)
            past_pressure_seq.append(env.pressure.copy())
            past_pos_w_seq.append(env.angle.copy())
            pred_speed = predictor.predict(past_pressure=np.array(past_pressure_seq), past_pos_w=np.array(past_pos_w_seq))
            print(colored(f"Predicted speed: {pred_speed}", "red"))
            print(colored(f"True speed: {env.speed}", "green"))
            if _is_predict:
                state[:3] = pred_speed
            # done = terminated or truncated
            done = False

            agent_position = env.agent_pos
            state_14 = env.state_14

            # Check subgoal success
            dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
            if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                done_lc = True
            t += 1
            ll_step += 1
            if done_lc:
                ppo_agent_lc.buffer.clear()
                break

        ppo_agent.buffer.clear()

    env.close()
    print("Total_Steps:", t)
    ##############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1, 2], help="Enter 0 to run test(), 1 to run success_rate_test(), 2 to run success_rate_test_1()")
    parser.add_argument("id", type=int, nargs="?", default=0, help="ID for success_rate_test() if mode is 1, Seed for success_rate_test_1() if mode is 2")
    args = parser.parse_args()

    if args.mode == 0:
        test()
    elif args.mode == 1:
        test_1()
    elif args.mode == 2:
        test_2()

