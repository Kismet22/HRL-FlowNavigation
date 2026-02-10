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
import math
import itertools
import pickle
import h5py

import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO
import hjbppo
from hjbppo import HJB_PPO
import icm_ppo
from icm_ppo import ICM_PPO


from env_new import train_env_basic_hovering

save_dir = './SuccessTest'


####### HRL Subgoal #######
def compute_subgoal_forward(pos, angle, r):
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
############################

########################### Mode 0: Fixed Target Test ###########################
def test_fixed(_id_in=1):
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 300
    # max_ep_len = max_steps + 20
    max_ep_len = max_steps
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    # state_dim_mode = 3
    # _is_normalize = False
    # ppo_path = f'./models/exp/lc/dim3/1231.pth'
    # _is_normalize = True
    # ppo_path = f'./models/exp/rl/dim3/9080.pth'

    # state_dim_mode = 11
    # _is_normalize = False
    # ppo_path = f'./models/exp/lc/dim11/1201.pth'
    # _is_normalize = True
    # ppo_path = f'./models/exp/rl/dim11/9071.pth'

    state_dim_mode = 6
    _is_normalize = False
    # ppo_path = f'./models/exp/lc_dim6.pth'
    # _is_normalize = False
    # ppo_path = f'./models/exp/lc/dim6/9210.pth'
    # _is_normalize = True
    ppo_path = f'./models/exp/rl/dim6/9083.pth'

    # set flow seed if required (0 : random flow; else : fixed flow)
    # flow_seed = 0
    # flow_seed = 15
    flow_seed = _id_in

    # set random mode(True : real random mode; else : random.seed(random_seed))
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    
    # test_mode = False
    test_mode = True
    switch_range = 4
    
    # switch_mode = True
    switch_mode = False

    input_max = 15.0
    clip_max = 15.0  

    ######################################################
    _start = np.array([float(180), float(32)])
    _target = np.array([float(180), float(32)])
    random_range = 32
    env = train_env_basic_hovering.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start, target_position=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize,
    _include_flow=True, _plot_flow=True, _proccess_flow=False,  
    _is_test = test_mode, _state_dim=state_dim_mode, video_dir="./Hovering_Experiment/video_frames",
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max)
    ######################################################

    ########################### State & Action ###########################
    # state space dimension
    # continuous action space; else discrete
    has_continuous_action_space = True
    state_dim = env.observation_space.shape[0]
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
    # Initialize Parameter
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
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)
    ######################################################

    ########################### Test ###########################
    flow_force_history = []
    action_force_history = []
    # 1 env reset
    state, info = env.reset()
    print("position_reset:", env.agent_pos)

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):
        # 2 env step
        action = ppo_agent.select_action(state)
        print(f"Step {i}, Action: {action}")
        state, reward, terminated, truncated, info = env.step(action)
        flow_force = env.f_fluid_history.copy()
        flow_force_history.append(flow_force)
        action_force = env.f_external_history.copy()
        action_force_history.append(action_force)
        # 3 save info
        ep_return_ppo += reward
        if env.d_2_target >= 24:
            break
        # if terminated or truncated:
        #     break
    env.close()
    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)
    np.save("./Experiment_flowforce/flow_force_history.npy", flow_force_history)
    np.save("./Experiment_flowforce/action_force_history.npy", action_force_history)
    print("Saved flow force and external force history.")
    ######################################################
###########################

########################### Mode 1: Hovering Test Controller###########################
def Hovering_test_HRL(_id_in=1):
    print("============================================================================================")

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    has_continuous_action_space = True
    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99  
    lr_actor = 0.0001
    lr_critic = 0.0002
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    _is_head = False
    video_dir = "./Video_HRL/video_frames"
    max_ll_steps = 5
    #####################################################  

    ##################################################### Tasks #####################################################  

    ########### 低速小范围正常任务 ###########
    _flow_info = True
    test_mode = True
    switch_range = 1
    theta_mode = False
    switch_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    random_range = 56
    max_steps = 300
    max_ep_len = max_steps + 20
    # max_ep_len = 200
    state_dim_mode = 10
    lc_state_dim = 6
    speed_range = 15.0
    speed_clip = 15.0
    # ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
    ppo_path_1 = f'./models/exp/lc_dim6.pth'
    ##################################################### ///// #####################################################

    ########### Environment ###########
    flow_seed = 1  
    # _start_position = np.array([64.0, 64.0])
    # _target_position = np.array([64.0, 64.0])
    _start_position = None
    _target_position = None
    env = train_env_upper_2.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
        _include_flow=_flow_info, _plot_flow=False, _proccess_flow=True, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=speed_range, v_range=speed_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov = _fov
    )

    ################ Action Space Settings ################
    state_dim = state_dim_mode
    expand_dim = state_dim
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    if has_continuous_action_space:
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    # ppo_agent = Classic_PPO(expand_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                         has_continuous_action_space,
    #                         action_std_init=action_std_4_test,
    #                         continuous_action_output_scale=action_output_scale,
    #                         continuous_action_output_bias=action_output_bias)
    # ppo_agent.load_full(ppo_path)
    # ppo_agent.set_eval_mode(True)

    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                               has_continuous_action_space,
                               action_std_init=action_std_4_test,
                               continuous_action_output_scale=lc_action_output_scale,
                               continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ppo_agent_lc.set_eval_mode(True)
    ###################################################### 

    ########################### Test ###########################
    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))

    # =====================================
    # 固定的流场编号与悬停测试点
    # =====================================
    flow_ids = [1, 3, 5, 10, 13, 15, 20, 23, 25, 27]
    hover_points = np.array([
        [180, 36],
        [180, 64],
        [180, 92],
        [240, 36],
        [240, 64],
        [240, 92]
    ])

    # 起点偏移半径和角度
    r_list = [0, 2, 4, 6]       # r=0 表示悬停点本身
    angles_deg = [45, 0, -45]   # 对 r>0 的情况

    # 保存路径
    save_dir = "./Experiment_hover_test_results"
    os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "hover_results.pkl")

    # ==== 原始 pickle 保存列表（保留） ====
    # results_all = []

    # ==== 新 HDF5 文件创建 ====
    h5_save_path = os.path.join(save_dir, "hover_results.h5")
    h5_file = h5py.File(h5_save_path, "w")

    for flow_id in flow_ids:
        for hover_point in hover_points:
            for r in r_list:
                angles = [0] if r == 0 else angles_deg  # r=0 时只用 angle=0
                for angle in angles:
                    # ==== 生成起点 ====
                    if r == 0:
                        start_pos = hover_point.copy()
                    else:
                        rad = np.deg2rad(angle)
                        offset = np.array([r * np.cos(rad), r * np.sin(rad)])
                        start_pos = hover_point + offset

                    # ==== 重置环境 ====
                    if _flow_info:
                        state, info, flow_info = env.reset(
                            _flow_init=int(flow_id),
                            _start_position_init=start_pos.astype(float).tolist(),
                            _target_position_init=hover_point.astype(float).tolist(),
                            _verbose=False
                        )
                    else:
                        state, info = env.reset(
                            _flow_init=int(flow_id),
                            _start_position_init=start_pos.astype(float).tolist(),
                            _target_position_init=hover_point.astype(float).tolist(),
                            _verbose=False
                        )

                    # ==== 初始化低层状态 ====
                    state_14 = env.state_14
                    agent_position = env.agent_pos
                    target_position = env.target_position

                    # ==== 悬停实验 ====
                    results = []
                    t = 0

                    while t < max_ep_len:
                        agent_position = env.agent_pos
                        lc_target_x, lc_target_y = target_position

                        # 构造低层状态
                        low_state = state_14.copy()
                        low_state[3] = lc_target_x - agent_position[0]
                        low_state[4] = lc_target_y - agent_position[1]

                        if lc_state_dim == 3:
                            low_state = low_state[3:6]
                        elif lc_state_dim == 6:
                            low_state = low_state[:6]
                        elif lc_state_dim == 11:
                            low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

                        # 低层动作
                        low_action = ppo_agent_lc.select_action(low_state)
                        if _flow_info:
                            state, reward, terminated, truncated, info, flow_info = env.step(low_action)
                        else:
                            state, reward, terminated, truncated, info = env.step(low_action)
                        agent_position = env.agent_pos
                        state_14 = env.state_14

                        # ==== 原记录每步数据（全部注释） ====
                        # results.append({
                        #     "step": t,
                        #     "position": agent_position.copy(),
                        #     "speed": env.speed,
                        #     "velocity": env.agent_velocity.copy(),
                        #     "agent_angle": env.angle,
                        #     "omega": env.vel_angle,
                        #     "action": low_action.copy()
                        # })

                        # ==== 新 HDF5 记录方式（只记录 t, state, action, flow_info） ====
                        step_record = {
                            "t": t,
                            "state": state.copy(),
                            "action": low_action.copy()
                        }
                        if _flow_info:
                            step_record["flow_info"] = flow_info.copy() if isinstance(flow_info, np.ndarray) else flow_info
                        results.append(step_record)

                        t += 1
                        if truncated:
                            break

                    # ==== 本轮轨迹结束后打印步数 ====
                    print(colored(f"Flow {flow_id}, Hover {hover_point}, r={r}, angle={angle} --> Total Steps: {t}", 'green'))

                    # ==== 原保存方式（pickle）全部注释 ====
                    # results_all.append({
                    #     "flow_id": int(flow_id),
                    #     "hover_point": hover_point.copy(),
                    #     "r": r,
                    #     "angle": angle,
                    #     "start_position": start_pos.copy(),
                    #     "trajectory": results
                    # })

                    # ==== HDF5 保存 ====
                    ep_name = f"flow{flow_id}_hover{hover_point[0]}_{hover_point[1]}_r{r}_ang{angle}"
                    grp = h5_file.create_group(ep_name)

                    t_arr = np.array([d["t"] for d in results], dtype=np.int32)
                    state_arr = np.array([d["state"] for d in results], dtype=np.float32)
                    action_arr = np.array([d["action"] for d in results], dtype=np.float32)

                    grp.create_dataset("t", data=t_arr, compression="gzip")
                    grp.create_dataset("state", data=state_arr, compression="gzip")
                    grp.create_dataset("action", data=action_arr, compression="gzip")

                    if _flow_info:
                        flow_arr = np.array([d["flow_info"] for d in results], dtype=np.float32)
                        grp.create_dataset("flow_info", data=flow_arr, compression="gzip")

                    # 元数据
                    grp.attrs["flow_id"] = int(flow_id)
                    grp.attrs["hover_point"] = hover_point.tolist()
                    grp.attrs["r"] = r
                    grp.attrs["angle"] = angle
                    grp.attrs["start_position"] = start_pos.tolist()

    # ==== 循环结束关闭环境和 HDF5 文件 ====
    env.close()
    h5_file.close()
    print(f"✅ Saved hover test results to HDF5: {h5_save_path}")

    # ==== 原 pickle 保存方式（保留，不删除） ====
    # with open(save_path, "wb") as f:
    #     pickle.dump(results_all, f)
    # print(f"✅ Saved hover test results: {save_path}")
###########################

########################### Mode 2: Hovering Test RL ###########################
def Hovering_test_RL(_id_in=1):
    print("============================================================================================")

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    has_continuous_action_space = True
    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99  
    lr_actor = 0.0001
    lr_critic = 0.0002
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    _is_head = False
    video_dir = "./Video_HRL/video_frames"
    max_ll_steps = 5
    #####################################################  

    ##################################################### Tasks #####################################################  

    ########### 低速小范围正常任务 ###########
    test_mode = True
    switch_range = 1
    theta_mode = False
    switch_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    random_range = 56
    max_steps = 200
    max_ep_len = max_steps + 20
    # max_ep_len = 200
    state_dim_mode = 10
    lc_state_dim = 6
    speed_range = 15.0
    speed_clip = 15.0
    ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
    ppo_path_1 = f'./models/exp/rl/dim6/9083.pth'
    ##################################################### ///// #####################################################

    ########### Environment ###########
    flow_seed = 1  
    # _start_position = np.array([64.0, 64.0])
    # _target_position = np.array([64.0, 64.0])
    _start_position = None
    _target_position = None
    env = train_env_upper_2.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
        # _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, 
        _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=speed_range, v_range=speed_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov = _fov
    )

    ################ Action Space Settings ################
    state_dim = state_dim_mode
    expand_dim = state_dim
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    if has_continuous_action_space:
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    ppo_agent = Classic_PPO(expand_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test,
                            continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)

    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                               has_continuous_action_space,
                               action_std_init=action_std_4_test,
                               continuous_action_output_scale=lc_action_output_scale,
                               continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ppo_agent_lc.set_eval_mode(True)
    ###################################################### 

    
    ########################### Test ###########################
    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    # =====================================
    # 固定的流场编号与悬停测试点
    # =====================================
    flow_ids = [1, 3, 5, 10, 13, 15, 20, 23, 25, 27]
    hover_points = np.array([
        [180, 36],
        [180, 64],
        [180, 92],
        [240, 36],
        [240, 64],
        [240, 92]
    ])

    # 起点偏移半径和角度
    r_list = [0, 2, 4, 6]       # r=0 表示悬停点本身
    angles_deg = [45, 0, -45]   # 对 r>0 的情况
    near_threshold = env.max_detect_dis + env.window_r/2

    # 保存路径
    save_dir = "./hover_test_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hover_results_rl.pkl")

    results_all = []

    for flow_id in flow_ids:
        for hover_point in hover_points:
            for r in r_list:
                angles = [0] if r == 0 else angles_deg  # r=0 时只用 angle=0
                for angle in angles:
                    # ==== 生成起点 ====
                    if r == 0:
                        start_pos = hover_point.copy()
                    else:
                        rad = np.deg2rad(angle)
                        offset = np.array([r * np.cos(rad), r * np.sin(rad)])
                        start_pos = hover_point + offset

                    # ==== 重置环境 ====
                    state, info = env.reset(
                        _flow_init=int(flow_id),  # 确保为 Python int
                        _start_position_init=start_pos.astype(float).tolist(),
                        _target_position_init=hover_point.astype(float).tolist(),
                        _verbose = False
                    )

                    # ==== 初始化低层状态 ====
                    state_14 = env.state_14
                    agent_position = env.agent_pos
                    target_position = env.target_position

                    # ==== 悬停实验 ====
                    results = []
                    t = 0

                    while t < max_ep_len:
                        agent_position = env.agent_pos
                        lc_target_x, lc_target_y = target_position

                        # 构造低层状态
                        low_state = state_14.copy()
                        low_state[3] = (lc_target_x - agent_position[0])/near_threshold
                        low_state[4] = (lc_target_y - agent_position[1])/near_threshold

                        if lc_state_dim == 3:
                            low_state = low_state[3:6]
                        elif lc_state_dim == 6:
                            low_state = low_state[:6]
                        elif lc_state_dim == 11:
                            low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

                        # 低层动作
                        low_action = ppo_agent_lc.select_action(low_state)
                        state, reward, terminated, truncated, info = env.step(low_action)
                        agent_position = env.agent_pos
                        state_14 = env.state_14

                        # 记录每步数据
                        results.append({
                            "step": t,
                            "position": agent_position.copy(),
                            "speed": env.speed,
                            "velocity": env.agent_velocity.copy(),
                            "agent_angle": env.angle,
                            "omega": env.vel_angle,
                            "action": low_action.copy()
                        })
                        t += 1
                        if truncated:
                            break
                    # ==== 本轮轨迹结束后打印步数 ====
                    print(colored(f"Flow {flow_id}, Hover {hover_point}, r={r}, angle={angle} --> Total Steps: {t}",'green'))
                    # ==== 保存每条轨迹 ====
                    results_all.append({
                        "flow_id": int(flow_id),
                        "hover_point": hover_point.copy(),
                        "r": r,
                        "angle": angle,
                        "start_position": start_pos.copy(),
                        "trajectory": results
                    })
    env.close()

    # ==== 保存 pickle ====
    with open(save_path, "wb") as f:
        pickle.dump(results_all, f)

    print(f"✅ Saved hover test results: {save_path}")
###########################

########################### Mode 3: Hovering Test HRL###########################
def Hovering_test_HRL_2(_id_in=1):
    print("============================================================================================")

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    has_continuous_action_space = True
    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99  
    lr_actor = 0.0001
    lr_critic = 0.0002
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    _is_head = False
    video_dir = "./Video_HRL/video_frames"
    max_ll_steps = 5
    #####################################################  

    ##################################################### Tasks #####################################################  

    ########### 低速小范围正常任务 ###########
    test_mode = True
    switch_range = 1
    theta_mode = False
    switch_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    random_range = 56
    max_steps = 200
    max_ep_len = max_steps + 20
    # max_ep_len = 200
    state_dim_mode = 6
    lc_state_dim = 6
    speed_range = 15.0
    speed_clip = 15.0
    ppo_path = f'./models/exp/hrl/dim6/9234.pth'
    ppo_path_1 = f'./models/exp/lc_dim6.pth'
    ##################################################### ///// #####################################################

    ########### Environment ###########
    flow_seed = 1  
    # _start_position = np.array([64.0, 64.0])
    # _target_position = np.array([64.0, 64.0])
    _start_position = None
    _target_position = None
    env = train_env_upper_2.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
        # _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, 
        _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=speed_range, v_range=speed_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov = _fov
    )

    ################ Action Space Settings ################
    state_dim = state_dim_mode
    expand_dim = state_dim
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    if has_continuous_action_space:
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    ppo_agent = Classic_PPO(expand_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test,
                            continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)

    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                               has_continuous_action_space,
                               action_std_init=action_std_4_test,
                               continuous_action_output_scale=lc_action_output_scale,
                               continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ppo_agent_lc.set_eval_mode(True)
    ###################################################### 

    
    ########################### Test ###########################
    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    # =====================================
    # 固定的流场编号与悬停测试点
    # =====================================
    flow_ids = [1, 3, 5, 10, 13, 15, 20, 23, 25, 27]
    hover_points = np.array([
        [180, 36],
        [180, 64],
        [180, 92],
        [240, 36],
        [240, 64],
        [240, 92]
    ])

    # 起点偏移半径和角度
    r_list = [0, 2, 4, 6]       # r=0 表示悬停点本身
    angles_deg = [45, 0, -45]   # 对 r>0 的情况

    # 保存路径
    save_dir = "./hover_test_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hover_results_planner.pkl")

    results_all = []
    subgoal_threshold = 1

    for flow_id in flow_ids:
        for hover_point in hover_points:
            for r in r_list:
                angles = [0] if r == 0 else angles_deg  # r=0 时只用 angle=0
                for angle in angles:
                    # ==== 生成起点 ====
                    if r == 0:
                        start_pos = hover_point.copy()
                    else:
                        rad = np.deg2rad(angle)
                        offset = np.array([r * np.cos(rad), r * np.sin(rad)])
                        start_pos = hover_point + offset

                    # ==== 重置环境 ====
                    state, info = env.reset(
                        _flow_init=int(flow_id),  # 确保为 Python int
                        _start_position_init=start_pos.astype(float).tolist(),
                        _target_position_init=hover_point.astype(float).tolist(),
                        _verbose = False
                    )

                    # ==== 初始化低层状态 ====
                    state_14 = env.state_14
                    agent_position = env.agent_pos
                    target_position = env.target_position

                    # ==== 悬停实验 ====
                    results = []
                    t = 0
                    r_sub = 2

                    while t < max_ep_len:
                        # === Get current high-level state ===
                        high_state = state
                        agent_position = env.agent_pos
                        use_fallback = env.d_2_target < switch_range
                        if use_fallback:
                            # Compute angle from agent to final target
                            vec_to_target = env.target_position - agent_position
                            angle_to_use = np.arctan2(vec_to_target[1], vec_to_target[0])
                        else:
                            # === Compute Subgoal ===
                            high_action = ppo_agent.select_action(high_state)
                            high_action = np.clip(high_action, env.low, env.high)
                            vec = high_action
                            vec = vec / (np.linalg.norm(vec) + 1e-6)
                            angle_to_use = np.arctan2(vec[1], vec[0])
                        lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=angle_to_use, r=r_sub)
                        # === Low-level loop ===
                        ll_step = 0
                        done_lc = False
                        while ll_step < max_ll_steps:
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
                            cliped_low_action = env.clip_action(low_action)

                            # Step env
                            state, reward, terminated, truncated, info = env.step(low_action)
                            done = terminated or truncated
                            agent_position = env.agent_pos
                            state_14 = env.state_14

                            # Check subgoal success
                            dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
                            if dist_to_subgoal < subgoal_threshold or truncated or ll_step + 1 >= max_ll_steps:
                                done_lc = True
                            # 记录每步数据
                            results.append({
                                "step": t,
                                "position": agent_position.copy(),
                                "speed": env.speed,
                                "velocity": env.agent_velocity.copy(),
                                "agent_angle": env.angle,
                                "omega": env.vel_angle,
                                "action": low_action.copy()
                                })
                            t += 1
                            ll_step += 1
                            if done_lc:
                                break
                        if truncated:
                            break
                    # ==== 本轮轨迹结束后打印步数 ====
                    print(colored(f"Flow {flow_id}, Hover {hover_point}, r={r}, angle={angle} --> Total Steps: {t}",'green'))
                    # ==== 保存每条轨迹 ====
                    results_all.append({
                        "flow_id": int(flow_id),
                        "hover_point": hover_point.copy(),
                        "r": r,
                        "angle": angle,
                        "start_position": start_pos.copy(),
                        "trajectory": results
                    })
    env.close()
    # ==== 保存 pickle ====
    with open(save_path, "wb") as f:
        pickle.dump(results_all, f)
    print(f"✅ Saved hover test results: {save_path}")
###########################

########################### Mode 4: Hovering Test Action ###########################
def Hovering_Action(_id_in=1):
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 300
    # max_ep_len = max_steps + 20
    max_ep_len = max_steps
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    _hrl_method = True

    # state_dim_mode = 3
    # if _hrl_method:
    #     _is_normalize = False
    #     ppo_path = f'./models/exp/lc/dim3/1231.pth'
    # else:
    #     _is_normalize = True
    #     ppo_path = f'./models/exp/rl/dim3/9080.pth'

    # state_dim_mode = 11
    # if _hrl_method:
    #     _is_normalize = False
    #     ppo_path = f'./models/exp/lc/dim11/1201.pth'
    # else:
    #     _is_normalize = True
    #     ppo_path = f'./models/exp/rl/dim11/9071.pth'

    state_dim_mode = 6
    if _hrl_method:
        _is_normalize = False
        # ppo_path = f'./models/exp/lc_dim6.pth'
        ppo_path = f'./models/exp/lc/dim6/9210.pth'
    else:
        _is_normalize = True
        ppo_path = f'./models/exp/rl/dim6/9083.pth'


    # set flow seed if required (0 : random flow; else : fixed flow)
    # flow_seed = 0
    flow_seed = _id_in

    # set random mode(True : real random mode; else : random.seed(random_seed))
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    
    # test_mode = False
    test_mode = True
    switch_range = 4
    
    # switch_mode = True
    switch_mode = False

    input_max = 15.0
    clip_max = 15.0  

    ######################################################
    _start = np.array([float(160), float(64)])
    _target = np.array([float(160), float(64)])
    random_range = 32
    env = train_env_basic_hovering.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start, target_position=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize,
    _include_flow=False, _plot_flow=False, _proccess_flow=False,  
    _is_test = test_mode, _state_dim=state_dim_mode, video_dir="./Hovering_Experiment/video_frames",
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max)
    ######################################################

    ########################### State & Action ###########################
    # state space dimension
    # continuous action space; else discrete
    has_continuous_action_space = True
    state_dim = env.observation_space.shape[0]
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
    # Initialize Parameter
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
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)
    ######################################################

    ########################### Test ###########################
    flow_ids = list(range(1, 26))
    hover_points = np.array([
        [180, 24],
        [180, 32],
        [180, 40],
        [180, 64],
        [180, 88],
        [180, 96],
        [180, 104],
        # [210, 36],
        # [210, 64],
        # [210, 92],
        # [240, 36],
        # [240, 64],
        # [240, 92]
    ])
    # r_list = [0, 2, 4, 6]
    # angles_deg = [45, 0, -45]
    r_list = [0]
    angles_deg = [0]

    # 保存路径
    save_dir = "./Experiment_hover_actions"
    os.makedirs(save_dir, exist_ok=True)

    # ==== 新 HDF5 文件创建 ====
    if _hrl_method:
        h5_save_path = os.path.join(save_dir, f"hrl_hover_dim_{state_dim_mode}.h5")
    else:
        h5_save_path = os.path.join(save_dir, f"rl_hover_dim_{state_dim_mode}.h5")
    h5_file = h5py.File(h5_save_path, "w")

    for flow_id in flow_ids:
        for hover_point in hover_points:
            for r in r_list:
                angles = [0] if r == 0 else angles_deg
                for angle in angles:
                    if r == 0:
                        start_pos = hover_point.copy()
                    else:
                        rad = np.deg2rad(angle)
                        offset = np.array([r * np.cos(rad), r * np.sin(rad)])
                        start_pos = hover_point + offset
                    state, info = env.reset(_flow_init=int(flow_id), _start_position_init=start_pos.astype(float).tolist(), _target_position_init=hover_point.astype(float).tolist(), _verbose=False)
                    agent_position = env.agent_pos
                    target_position = env.target_position

                    # ==== episode ====
                    results = []
                    success = False  # 初始化成功标记
                    for i in range(1, max_ep_len + 1):
                        # 选择动作
                        action = ppo_agent.select_action(state)

                        # 环境更新
                        state, reward, terminated, truncated, info = env.step(action)

                        # 保存受力
                        agent_position = env.agent_pos.copy()
                        flow_force = env.f_fluid_history.copy()
                        action_force = env.f_external_history.copy()
                        agent_velocity = env.agent_velocity.copy()
                        agent_angle = env.angle.copy()
                        agent_omega = env.vel_angle.copy()
                        step_record = {
                            "step": i,
                            "position": agent_position,
                            "velocity": agent_velocity,
                            "agent_angle": agent_angle,
                            "omega": agent_omega,
                            "flow_force": flow_force,
                            "action_force": action_force
                        }
                        results.append(step_record)

                        # 检查目标条件
                        if env.d_2_target >= 16:  # 提前离开目标，失败
                            success = False
                            break
                    else:
                        # 如果循环没有提前 break，说明到达 max_ep_len，算成功
                        success = True

                    print(colored(
                        f"Flow {flow_id}, Hover {hover_point}, r={r}, angle={angle} --> Total Steps: {i}, Success: {success}",
                        'green' if success else 'red'  # success=True -> green, False -> red
                    ))

                    # ==== HDF5 保存（只保存力） ====
                    ep_name = f"flow{flow_id}_hover{hover_point[0]}_{hover_point[1]}_r{r}_ang{angle}"
                    grp = h5_file.create_group(ep_name)

                    # 转换为 numpy 数组
                    steps_arr = np.array([d["step"] for d in results], dtype=np.int32)
                    flow_force_arr = np.array([d["flow_force"] for d in results], dtype=np.float32)
                    action_force_arr = np.array([d["action_force"] for d in results], dtype=np.float32)
                    position_arr = np.array([d["position"] for d in results], dtype=np.float32)
                    velocity_arr = np.array([d["velocity"] for d in results], dtype=np.float32)
                    angle_arr = np.array([d["agent_angle"] for d in results], dtype=np.float32)
                    omega_arr = np.array([d["omega"] for d in results], dtype=np.float32)

                    # 创建 dataset
                    grp.create_dataset("step", data=steps_arr, compression="gzip")
                    grp.create_dataset("flow_force", data=flow_force_arr, compression="gzip")
                    grp.create_dataset("action_force", data=action_force_arr, compression="gzip")
                    grp.create_dataset("position", data=position_arr, compression="gzip")
                    grp.create_dataset("velocity", data=velocity_arr, compression="gzip")
                    grp.create_dataset("agent_angle", data=angle_arr, compression="gzip")
                    grp.create_dataset("omega", data=omega_arr, compression="gzip")


                    # 成功标记保存为属性
                    grp.attrs["success"] = int(success)  # 1 = 成功, 0 = 失败

                    # 元数据
                    grp.attrs["flow_id"] = int(flow_id)
                    grp.attrs["hover_point"] = hover_point.tolist()
                    grp.attrs["r"] = r
                    grp.attrs["angle"] = angle
                    grp.attrs["start_position"] = start_pos.tolist()

    # ==== 循环结束关闭环境和 HDF5 文件 ====
    env.close()
    h5_file.close()
    print(f"✅ Saved hover test results to HDF5: {h5_save_path}")
###########################

########################### Mode 5: Hovering Test Action Direction ###########################
def Hovering_Direction(_id_in=1):
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 1500
    # max_ep_len = max_steps + 20
    max_ep_len = max_steps
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    # state_dim_mode = 3
    # ppo_path = f'./models/exp/rl/dim3/9080.pth'

    # state_dim_mode = 11
    # ppo_path = f'./models/exp/rl/dim11/9071.pth'

    state_dim_mode = 6
    _hrl_method = False
    if _hrl_method:
        _is_normalize = False
        ppo_path = f'./models/exp/lc_dim6.pth'
    else:
        _is_normalize = True
        ppo_path = f'./models/exp/rl/dim6/9083.pth'


    # set flow seed if required (0 : random flow; else : fixed flow)
    # flow_seed = 0
    # flow_seed = 15
    flow_seed = _id_in

    # set random mode(True : real random mode; else : random.seed(random_seed))
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    
    # test_mode = False
    test_mode = True
    switch_range = 4
    
    # switch_mode = True
    switch_mode = False

    input_max = 15.0
    clip_max = 15.0  

    ######################################################
    _start = np.array([float(160), float(64)])
    _target = np.array([float(160), float(64)])
    random_range = 32
    env = train_env_basic_hovering.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start, target_position=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize,
    _include_flow=False, _plot_flow=False, _proccess_flow=False,  
    _is_test = test_mode, _state_dim=state_dim_mode, video_dir="./Hovering_Experiment/video_frames",
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max)
    ######################################################

    ########################### State & Action ###########################
    # state space dimension
    # continuous action space; else discrete
    has_continuous_action_space = True
    state_dim = env.observation_space.shape[0]
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
    # Initialize Parameter
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
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)
    ######################################################

    ########################### Test : Trajectory Tracking ###########################
    flow_ids = list(range(1, 26))

    trajectory_list = []
    trajectory_names = []

    # ========= 轨迹参数 =========
    ds = 2.0
    n_points = 250  # 圆形轨迹点数
    r_circle = 32
    center_x, center_y = 200, 64

    # ---- 顺时针/逆时针轨迹生成函数 ----
    def generate_circle_trajectory(center_x, center_y, start_x, start_y, n_points=250, clockwise=True):
        theta0 = np.arctan2(start_y - center_y, start_x - center_x)
        t = np.linspace(0, 2*np.pi, n_points)
        if clockwise:
            angles = -t + theta0
        else:
            angles = t + theta0
        x = center_x + r_circle * np.cos(angles)
        y = center_y + r_circle * np.sin(angles)
        return np.vstack((x, y)).T

    # ========= 生成四条圆形轨迹 =========
    circle_points = [
        (232, 64, True),   # 起点232,64 顺时针
        (232, 64, False),  # 起点232,64 逆时针
        (168, 64, True),   # 起点168,64 顺时针
        (168, 64, False),  # 起点168,64 逆时针
    ]

    for start_x, start_y, clockwise in circle_points:
        traj = generate_circle_trajectory(center_x, center_y, start_x, start_y, n_points, clockwise)
        trajectory_list.append(traj)
        direction = "cw" if clockwise else "ccw"
        trajectory_names.append(f"circle_r{r_circle}_start_{start_x}_{start_y}_{direction}")

    # ========= 保存路径 =========
    save_dir = "./Experiment_tracking_directions_force"
    os.makedirs(save_dir, exist_ok=True)

    if _hrl_method:
        h5_save_path = os.path.join(save_dir, "hrl_tracking_force.h5")
    else:
        h5_save_path = os.path.join(save_dir, "rl_tracking_force.h5")

    h5_file = h5py.File(h5_save_path, "w")

    # ============================================================================
    #                               主循环
    # ============================================================================
    for flow_id in flow_ids:
        for traj_points, traj_name in zip(trajectory_list, trajectory_names):

            # ===== reset 环境（起点 + 第一个目标点）=====
            state, info = env.reset(
                _flow_init=int(flow_id),
                _start_position_init=np.array(traj_points[0], dtype=np.float32),
                _target_position_init=np.array(traj_points[1], dtype=np.float32),
                _verbose=False
            )

            traj_idx = 1
            t_step = 0
            done = False
            success = False

            results = []

            # ================= episode =================
            while not done and t_step < max_ep_len:

                # ---- 选择动作 ----
                action = ppo_agent.select_action(state)

                # ---- step ----
                state, reward, terminated, truncated, info = env.step(action)

                step_record = {
                    "step": t_step,
                    "traj_idx": traj_idx,
                    "position": env.agent_pos.copy(),
                    "velocity": env.agent_velocity.copy(),
                    "flow_force": env.f_fluid_history.copy(),
                    "action_force": env.f_external_history.copy()
                }
                results.append(step_record)

                t_step += 1

                # ---- 当前子目标完成 ----
                if terminated:
                    traj_idx += 1

                    if traj_idx < len(traj_points):
                        env.set_target(np.array(traj_points[traj_idx], dtype=np.float32))
                    else:
                        success = True
                        done = True

                if truncated:
                    success = False
                    break

            print(colored(
                f"[Flow {flow_id}] {traj_name} --> Steps: {t_step}, Success: {success}",
                "green" if success else "red"
            ))

            # ================= HDF5 保存（安全版） =================
            ep_name = f"{traj_name}_flow{flow_id}"

            if ep_name in h5_file:
                grp = h5_file[ep_name]
                print(f"[Warning] Group {ep_name} already exists, using existing group.")
            else:
                grp = h5_file.create_group(ep_name)

            steps_arr = np.array([d["step"] for d in results], dtype=np.int32)
            traj_idx_arr = np.array([d["traj_idx"] for d in results], dtype=np.int32)
            position_arr = np.array([d["position"] for d in results], dtype=np.float32)
            velocity_arr = np.array([d["velocity"] for d in results], dtype=np.float32)
            flow_force_arr = np.array([d["flow_force"] for d in results], dtype=np.float32)
            action_force_arr = np.array([d["action_force"] for d in results], dtype=np.float32)

            for dname, data in zip(
                ["step","traj_idx","position","velocity","flow_force","action_force"],
                [steps_arr, traj_idx_arr, position_arr, velocity_arr, flow_force_arr, action_force_arr]
            ):
                if dname in grp:
                    del grp[dname]
                grp.create_dataset(dname, data=data, compression="gzip")

            grp.attrs["success"] = int(success)
            grp.attrs["flow_id"] = int(flow_id)
            grp.attrs["trajectory_name"] = traj_name
            grp.attrs["n_points"] = len(traj_points)
            grp.attrs["ds"] = ds

    # ============================================================================
    env.close()
    h5_file.close()
    print(f"✅ Saved tracking force results to HDF5: {h5_save_path}")
###########################

########################### Mode 6: Hovering Test Action Direction (Straight Line) ###########################
def Hovering_Direction_Line(_id_in=1):
    print("============================================================================================")
    ########################### Environment ###########################
    max_steps = 400
    # max_ep_len = max_steps + 20
    max_ep_len = max_steps
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    state_dim_mode = 6
    _hrl_method = True
    if _hrl_method:
        _is_normalize = False
        ppo_path = f'./models/exp/lc_dim6.pth'
    else:
        _is_normalize = True
        ppo_path = f'./models/exp/rl/dim6/9083.pth'

    flow_seed = _id_in
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    
    test_mode = True
    switch_range = 4
    switch_mode = False
    input_max = 15.0
    clip_max = 15.0  

    _start = np.array([float(160), float(64)])
    _target = np.array([float(160), float(64)])
    random_range = 32
    env = train_env_basic_hovering.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start, target_position=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize,
    _include_flow=False, _plot_flow=False, _proccess_flow=False,  
    _is_test = test_mode, _state_dim=state_dim_mode, video_dir="./Hovering_Experiment/video_frames",
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max)

    ########################### State & Action ###########################
    has_continuous_action_space = True
    state_dim = env.observation_space.shape[0]
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n

    ########################### PPO ###########################
    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0001
    lr_critic = 0.0002
    action_std_4_test = 0.04

    ppo_agent = Classic_PPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma,
        K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test,
        continuous_action_output_scale=action_output_scale,
        continuous_action_output_bias=action_output_bias
    )
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)

    ########################### Test : Straight Line Trajectory Tracking ###########################
    flow_ids = list(range(1, 26))
    trajectory_list = []
    trajectory_names = []

    # ========= 直线轨迹参数 =========
    ds = 2.0        # 点间距
    line_length = 60  # 直线长度
    start_center = np.array([240, 64], dtype=np.float32)
    angles_deg = [120, 135, 180, 225, 240]  # 直线方向

    def generate_line_trajectory(start, angle_deg, ds=2.0, length=100):
        theta = np.deg2rad(angle_deg)
        n_points = int(length / ds)
        x = start[0] + np.arange(n_points) * ds * np.cos(theta)
        y = start[1] + np.arange(n_points) * ds * np.sin(theta)
        return np.vstack((x, y)).T

    for angle in angles_deg:
        traj = generate_line_trajectory(start_center, angle, ds=ds, length=line_length)
        trajectory_list.append(traj)
        trajectory_names.append(f"line_angle_{angle}")

    # ========= HDF5 保存 =========
    save_dir = "./Experiment_tracking_directions_force_line"
    os.makedirs(save_dir, exist_ok=True)
    if _hrl_method:
        h5_save_path = os.path.join(save_dir, "hrl_tracking_force_line.h5")
    else:
        h5_save_path = os.path.join(save_dir, "rl_tracking_force_line.h5")
    h5_file = h5py.File(h5_save_path, "w")

    # ========================================================================
    #                               主循环
    # ========================================================================
    for flow_id in flow_ids:
        for traj_points, traj_name in zip(trajectory_list, trajectory_names):
            state, info = env.reset(
                _flow_init=int(flow_id),
                _start_position_init=np.array(traj_points[0], dtype=np.float32),
                _target_position_init=np.array(traj_points[1], dtype=np.float32),
                _verbose=False
            )

            traj_idx = 1
            t_step = 0
            done = False
            success = False
            results = []

            while not done and t_step < max_ep_len:
                action = ppo_agent.select_action(state)
                state, reward, terminated, truncated, info = env.step(action)

                step_record = {
                    "step": t_step,
                    "traj_idx": traj_idx,
                    "position": env.agent_pos.copy(),
                    "velocity": env.agent_velocity.copy(),
                    "flow_force": env.f_fluid_history.copy(),
                    "action_force": env.f_external_history.copy()
                }
                results.append(step_record)
                t_step += 1

                if terminated:
                    traj_idx += 1
                    if traj_idx < len(traj_points):
                        env.set_target(np.array(traj_points[traj_idx], dtype=np.float32))
                    else:
                        success = True
                        done = True
                if truncated:
                    success = False
                    break

            print(colored(
                f"[Flow {flow_id}] {traj_name} --> Steps: {t_step}, Success: {success}",
                "green" if success else "red"
            ))

            # ================= HDF5 保存 =================
            ep_name = f"{traj_name}_flow{flow_id}"
            if ep_name in h5_file:
                grp = h5_file[ep_name]
            else:
                grp = h5_file.create_group(ep_name)

            steps_arr = np.array([d["step"] for d in results], dtype=np.int32)
            traj_idx_arr = np.array([d["traj_idx"] for d in results], dtype=np.int32)
            position_arr = np.array([d["position"] for d in results], dtype=np.float32)
            velocity_arr = np.array([d["velocity"] for d in results], dtype=np.float32)
            flow_force_arr = np.array([d["flow_force"] for d in results], dtype=np.float32)
            action_force_arr = np.array([d["action_force"] for d in results], dtype=np.float32)

            for dname, data in zip(
                ["step","traj_idx","position","velocity","flow_force","action_force"],
                [steps_arr, traj_idx_arr, position_arr, velocity_arr, flow_force_arr, action_force_arr]
            ):
                if dname in grp:
                    del grp[dname]
                grp.create_dataset(dname, data=data, compression="gzip")

            grp.attrs["success"] = int(success)
            grp.attrs["flow_id"] = int(flow_id)
            grp.attrs["trajectory_name"] = traj_name
            grp.attrs["n_points"] = len(traj_points)
            grp.attrs["ds"] = ds

    env.close()
    h5_file.close()
    print(f"✅ Saved straight line tracking results to HDF5: {h5_save_path}")
###########################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1, 2, 3, 4, 5, 6], help="Enter 0 to run fixed point test; 1,2,3 to run multi-test")
    parser.add_argument("id", type=int, nargs="?", default=0, help="Random seed for the test function (0 for fully random, otherwise fixed)")
    args = parser.parse_args()

    if args.mode == 0:
        test_fixed(args.id)
    elif args.mode == 1:
        Hovering_test_HRL(args.id)
    elif args.mode == 2:
        Hovering_test_RL(args.id)
    elif args.mode == 3:
        Hovering_test_HRL2(args.id)
    elif args.mode == 4:
        Hovering_Action(args.id)
    elif args.mode == 5:
        Hovering_Direction(args.id)
    elif args.mode == 6:
        Hovering_Direction_Line(args.id)