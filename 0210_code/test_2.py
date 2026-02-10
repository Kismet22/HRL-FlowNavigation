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


from env_new import train_env_upper_2
from env_new import train_env_plus
from env_new import train_env_upper_2_1
from env_new import train_env_basic_2_2
from env_new import train_env_upper_2_3


save_dir = './SuccessTest'
save_dir_1 = './model_output'


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

####### Read Point #######
def read_and_check(file_path):
    """ read .csv flie,convert to numpy array """
    if not os.path.exists(file_path):
        print(f"File {file_path} Not Exist!")
        return None, 0
    try:
        df = pd.read_csv(file_path)
        df_numpy = df.to_numpy()
        _length = df.shape[0]
        return df_numpy, _length
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, 0
############################

########################### Mode 0: HRL Single Test ###########################
def test(_id_in=1):
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
    true_random = True
    # true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    _is_head = False
    video_dir = "./10_Video_HRL_Avoidance/video_frames"
    max_ll_steps = 5
    #####################################################  

    ##################################################### Tasks #####################################################  

    ########### 低速小范围正常任务 ###########
    flow_seed = _id_in
    # flow_seed = 1 
    # test_mode = False
    test_mode = True
    theta_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    switch_mode = False



    _start_position = None
    _target_position = None
    force_range = 15.0
    force_clip = 15.0

    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])

    # switch_mode = True
    # _start = np.array([240.0, 64.0])
    # _target = np.array([160.0, 64.0])

    # random_range = 32
    random_range = 56
    max_steps = 600
    max_ep_len = max_steps + 20

    state_dim_mode = 10
    _model_id = 1
    # state_dim_mode = 8
    # _model_id = 3
    ppo_path = f'./models/exp/hrl/dim{state_dim_mode}/a15+fov180_{_model_id}.pth'
    switch_range = 8
    # state_dim_mode = 6
    # _model_id = 1
    # ppo_path = f'./models/exp/hrl/dim{state_dim_mode}/{_model_id}.pth'
    # switch_range = 8
    
    lc_state_dim = 6
    ppo_path_1 = f'./models/exp/lc_dim6.pth'
    video_dir = "./EXP_HRL/video_frames"
    # env = train_env_upper_2.foil_env(
    #     args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position = _start_position, target_position = _target_position,
    #     _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, 
    #     _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
    #     _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
    #     u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
    #     _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
    #     _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov
    # )

    env = train_env_plus.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=None, target_position=None,
        _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov
    )
    
    ################ Action Space Settings ################
    state_dim = env.observation_space.shape[0]
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

    subgoal_threshold = 1
    x_start = 220
    y_start = 32
    x_target = 64
    y_target = 64
    _extra_obs = True

    _start_position_init = np.array([x_start, y_start])
    _target_position_init = np.array([x_target, y_target])
    state, info = env.reset(_start_position_init=_start_position_init, _target_position_init=_target_position_init, _add_virtual_cylinders=_extra_obs, _custom_cylinders=[(180, 64), (180, 108), (180, 16)])

    # _start_position_init = np.array([220, 32])
    # _target_position_init = np.array([64, 64])
    # # _start_position_init = np.array([float(173), float(81)])
    # # _target_position_init = np.array([float(19.3), float(33.3)])
    # state, info = env.reset(_start_position_init=_start_position_init, _target_position_init=_target_position_init)

    # Store initial states
    state_14 = env.state_14
    agent_position = env.agent_pos
    prev_state = state.copy()
    prev_state_14 = state_14.copy()
    prev_agent_position = agent_position.copy()
    print("position_reset:", agent_position)
    target_position = env.target_position

    # === 初始化记录容器 ===
    record_high_states = []
    record_positions = []

    # Main episode loop
    t = 0
    hl_time_step = 0
    done = False

    while not done and t < max_ep_len:
        # === Get current high-level state ===
        high_state = state

        # === 记录当前高层状态和智能体位置 ===
        record_high_states.append(high_state.copy())
        record_positions.append(agent_position.copy())

        # === Decide whether to use fallback or normal high-level policy ===
        use_fallback = env.d_2_target < switch_range
        if use_fallback:
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
            r = np.linalg.norm(env.target_position - agent_position) if use_fallback else 2
            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=angle_to_use, r=r)
        else:
            if use_fallback:
                r = np.linalg.norm(env.target_position - agent_position)
                angle_to_use = target_angle
            else:
                r = 2
                if not theta_mode:
                    vec = high_action / (np.linalg.norm(high_action) + 1e-6)
                    angle_to_use = np.arctan2(vec[1], vec[0])
                else:
                    angle_to_use = high_action
            lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=angle_to_use, r=r)

        env.set_lc_target([lc_target_x, lc_target_y])
        print(f"=== Subgoal: ({lc_target_x:.2f}, {lc_target_y:.2f}) | Angle used: {angle_to_use:.2f} ===")

        # === Low-level loop ===
        ll_step = 0
        done_lc = False
        while ll_step < max_ll_steps:
            agent_angle_old = env.angle

            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]

            if lc_state_dim == 3:
                low_state = low_state[3:6]
            elif lc_state_dim == 6:
                low_state = low_state[:6]
            elif lc_state_dim == 11:
                low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

            low_action = ppo_agent_lc.select_action(low_state)
            cliped_low_action = env.clip_action(low_action)
            print(f"=== Low Level Step {t}, Action: {low_action}")
            print(f"===               -> Clip_Action: {cliped_low_action} ===")

            state, reward, terminated, truncated, info = env.step(low_action)
            done = terminated or truncated
            agent_position = env.agent_pos
            state_14 = env.state_14

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

    # === 保存记录结果 ===
    save_dir = "./Experiment/Single_test/"
    os.makedirs(save_dir, exist_ok=True)   # 自动创建目录（若已存在则跳过）

    record_high_states = np.array(record_high_states)
    record_positions = np.array(record_positions)

    np.save(os.path.join(save_dir, f"record_high_states_dim_{state_dim_mode}.npy"), record_high_states)
    np.save(os.path.join(save_dir, f"record_positions_dim_{state_dim_mode}.npy"), record_positions)

    print(f"Saved: {save_dir}record_high_states_dim_{state_dim_mode}.npy")
    print(f"Saved: {save_dir}record_positions_dim_{state_dim_mode}.npy")
    ###########################
###########################

########################### Mode 1: Multi-Test ###########################
def success_rate_test_HRL(_id_in=1):
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
    np.random.seed(random_seed)
    #####################################################  

    ################ General Settings ################
    # -------- Save & Debug --------
    _save_traj = True  # If True, save agent trajectories for visualization.
    _save_state = False  # If True, save full environment states for detailed analysis.
    _extra_obs = False  # If True, introduce additional obstacles into the environment.

    # -------- Test / Flow --------
    if _save_traj:
        test_mode = True
        flow_seed = 1
    else:
        test_mode = False
        flow_seed = 0 # random flow id

    switch_range = 8  # Distance threshold for determining whether the agent is close to the target.
    theta_mode  = False  # If True, the high-level policy outputs a heading angle.
    switch_mode = False  # If True, the start and target regions are swapped.

    # -------- Observation / Control --------
    _is_normalize = True
    _is_near = False  # If True, start and target positions are selected near obstacles.
    _fov = np.pi  # Field of view for obstacle observation (in radians).

    # -------- Speed --------
    force_range = 15.0
    force_clip  = 15.0

    # -------- Low-level --------
    lc_state_dim = 6
    ppo_path_lc  = "./models/exp/lc_dim6.pth"
    #####################################################

    ################ Task Settings ################
    TASK_MODE = "avoid"      # "avoid" | "free"

    if TASK_MODE == "avoid":
        """
        Obstacle avoidance task
        """
        _is_avoid = True

        _start  = np.array([220.0, 64.0]) # train [180, 64]
        _target = np.array([64.0,  64.0])

        random_range = 56 # train range: 32
        max_steps = 600 # no extra obs train: 360
        r_test = 2

        # ---- State / Model ----
        state_dim_mode = 10 # 6:no obs 8:no guidance 10:with guidance
        model_id = 1

        ppo_path = (
            f"./models/exp/hrl/"
            f"dim{state_dim_mode}/"
            f"a15+fov180_{model_id}.pth"
        )

    elif TASK_MODE == "free":
        """
        Free navigation task (no obstacle)
        """
        _is_avoid = False

        _start  = np.array([240.0, 64.0])
        _target = np.array([160.0, 64.0])

        random_range = 32
        max_steps = 200
        r_test = 2

        # ---- State / Model ----
        state_dim_mode = 6
        model_id = 5

        ppo_path = (
            f"./models/exp/hrl/"
            f"dim{state_dim_mode}/"
            f"{model_id}.pth"
        )

    else:
        raise ValueError(f"Unknown TASK_MODE: {TASK_MODE}")

    # ================= Derived Parameters ====================
    max_ep_len = max_steps + 20
    _start_position  = None
    _target_position = None
    #####################################################

    ########### high speed test ###########
    # flow_seed = 0 
    # test_mode = False
    # switch_range = 4
    # theta_mode = False
    # switch_mode = False
    # _is_normalize = True
    # _is_avoid = True
    # _is_near = True
    # _fov = 1.5 * np.pi
    # _start = np.array([220.0, 64.0])
    # _target = np.array([64.0, 64.0])
    # random_range = 32
    # max_steps = 240
    # max_ep_len = max_steps + 20
    # state_dim_mode = 10
    # lc_state_dim = 14
    # force_range = 20.0
    # force_clip = 20.0
    # ppo_path = f'./models/exp/hrl/dim10/a20+fov270+near.pth'
    # ppo_path_1 = f'./models/exp/lc_dim14_a20.pth'
    ##################################################### ///// ##################################################### 

    
    ########### Environment ########### 
    # env = train_env_upper_2.foil_env(
    #     args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
    #     _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, 
    #     _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
    #     _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
    #     u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
    #     _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
    #     _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov = _fov
    # )

    env = train_env_plus.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
        _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov
    )
    ################ Action Space Settings ################
    state_dim = env.observation_space.shape[0]
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
    ppo_agent_lc.load_full(ppo_path_lc)
    ppo_agent_lc.set_eval_mode(True)
    ######################################################

    ########################### Test ###########################
    subgoal_threshold = 1
    test_time = 20 if _save_traj else 2000
    success_times = 0
    collide_times = 0
    timeout_times = 0
    outbound_times = 0
    count_time = []

    # print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    print(colored(f"State dimension: {state_dim_mode} | Model ID: {model_id}", 'green'))
    save_items = []
    if _save_traj:
        save_items.append("trajectory (pickle)")
    if _save_state:
        save_items.append("state (HDF5)")

    if save_items:
        print(colored(f"Will save: {', '.join(save_items)}", 'yellow'))
    else:
        print(colored("No data will be saved.", 'yellow'))
    
    # === CSV 文件初始化（保留原来的 episode 统计） ===
    csv_path = f"./Experiment/HRL_a15_test_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Start_X", "Start_Y", "Target_X", "Target_Y", "Flow_ID", "End_State", "End_angle", "End_time"])

    # === Trajectory pickle 初始化 ===
    if _save_traj:
        traj_pickle_path = f"./Experiment/HRL_trajectories_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}.pkl"
        os.makedirs(os.path.dirname(traj_pickle_path), exist_ok=True)

    # === HDF5 文件初始化，用于 _save_state ===
    if _save_state:
        state_h5_path = f"./Experiment/HRL_states_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}.h5"
        os.makedirs(os.path.dirname(state_h5_path), exist_ok=True)
        state_h5_file = h5py.File(state_h5_path, "a")

    # ==== 主测试循环 ====
    for n in range(test_time):
        # ==== Env Reset ====
        # center 200
        # x_start = 260.0
        # x_target = 140.0
        # y_start = np.random.uniform(32, 96)
        # y_target = np.random.uniform(32, 96)
        # _start_position_init = np.array([x_start, y_start])
        # _target_position_init = np.array([x_target, y_target])
        # state, info = env.reset(_start_position_init=_start_position_init, _target_position_init=_target_position_init, _add_virtual_cylinders=_extra_obs, _custom_cylinders=[(200, 64), (200, 108), (200, 16)])

        state, info = env.reset(_add_virtual_cylinders=_extra_obs, _custom_cylinders=[(160, 64), (160, 108), (160, 16)])
        state_14 = env.state_14
        agent_position = env.agent_pos
        start_position = np.copy(agent_position)
        flow_id = env.flow_init

        episode_traj = []    # traj saving
        episode_state = []   # state saving
        results = []

        t = 0
        hl_time_step = 0
        done = False
        while not done and t < max_ep_len:
            # ==== High-level policy ====
            high_state = state
            use_fallback = env.d_2_target < switch_range
            if use_fallback:
                vec_to_target = env.target_position - agent_position
                target_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            else:
                high_action = ppo_agent.select_action(high_state)
                high_action = np.clip(high_action, env.low, env.high)
                hl_time_step += 1

            # ==== Planner Policy ====
            agent_position = env.agent_pos
            if _is_head:
                angle_to_use = target_angle if use_fallback else high_action
                r = np.linalg.norm(env.target_position - agent_position) if use_fallback else r_test
                lc_target_x, lc_target_y = compute_subgoal_forward(agent_position, angle_to_use, r)
            else:
                r = np.linalg.norm(env.target_position - agent_position) if use_fallback else r_test
                if use_fallback:
                    angle_to_use = target_angle
                else:
                    vec = high_action / (np.linalg.norm(high_action) + 1e-6) if not theta_mode else high_action
                    angle_to_use = np.arctan2(vec[1], vec[0]) if not theta_mode else high_action
                lc_target_x, lc_target_y = compute_subgoal(agent_position, angle_to_use, r)

            env.set_lc_target([lc_target_x, lc_target_y])
            # ==== Low-level loop ====
            ll_step = 0
            done_lc = False
            while ll_step < max_ll_steps:
                # ==== 构造低层状态 ====
                low_state = state_14.copy()
                low_state[3] = lc_target_x - agent_position[0]
                low_state[4] = lc_target_y - agent_position[1]

                if lc_state_dim == 3:
                    low_state = low_state[3:6]
                elif lc_state_dim == 6:
                    low_state = low_state[:6]
                elif lc_state_dim == 11:
                    low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

                # ==== 低层动作选择 ====
                low_action = ppo_agent_lc.select_action(low_state)
                cliped_low_action = env.clip_action(low_action)

                # ==== 执行动作更新环境 ====
                state, reward, terminated, truncated, info = env.step(low_action)
                done = terminated or truncated
                agent_position = env.agent_pos
                state_14 = env.state_14

                # ==== 保存 state（HDF5） ====
                if _save_state:
                    # 保存 t, state, low_state, high_action, low_action
                    episode_state.append({
                        "t": t,
                        "state": state.copy(),
                        "low_state": low_state.copy(),
                        "high_action": high_action.tolist() if high_action is not None else None,
                        "low_action": low_action.tolist() if hasattr(low_action, 'tolist') else low_action
                    })
                # ==== 保存轨迹数据 ====
                if _save_traj:
                    episode_traj.append({
                        "step": t,
                        "agent_x": agent_position[0], "agent_y": agent_position[1],
                        "lc_target_x": lc_target_x, "lc_target_y": lc_target_y,
                        "high_action": None if use_fallback else high_action.tolist() if hasattr(high_action, 'tolist') else high_action, "low_action": low_action.tolist() if hasattr(low_action, 'tolist') else low_action,
                        "reward": reward, "done": done
                    })

                # ==== 距离子目标判断 ====
                dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
                if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                    done_lc = True

                t += 1
                ll_step += 1
                if done_lc:
                    break
        # ==== Episode 结束 ====
        end_state = env.end_state
        end_angle = env.angle
        start_x, start_y = start_position
        target_x, target_y = env.target_position
        results.append([n, start_x, start_y, target_x, target_y, flow_id, end_state, end_angle, t])

        # ==== 保存 pickle 轨迹 ====
        if _save_traj:
            with open(traj_pickle_path, "ab") as f:
                pickle.dump(episode_traj, f)
                f.flush()
                os.fsync(f.fileno())

        # ==== 保存 HDF5 state 文件 ====
        if _save_state and len(episode_state) > 0:
            grp_name = f"episode_{n:05d}"
            if grp_name in state_h5_file:
                del state_h5_file[grp_name]
            grp = state_h5_file.create_group(grp_name)

            grp.create_dataset("t", data=np.array([s["t"] for s in episode_state]), compression="gzip")
            grp.create_dataset("state", data=np.array([s["state"] for s in episode_state]), compression="gzip")
            grp.create_dataset("low_state", data=np.array([s["low_state"] for s in episode_state]), compression="gzip")

            # high_action 可能为 None
            for s in episode_state:
                if s["high_action"] is not None:
                    high_action_dim = len(s["high_action"])
                    break
            else:
                high_action_dim = 2  # 默认值
            high_action_data = np.array([
                s["high_action"] if s["high_action"] is not None else [np.nan]*high_action_dim
                for s in episode_state
            ])
            grp.create_dataset("high_action", data=high_action_data, compression="gzip")

            # low_action
            low_action_data = np.array([
                np.atleast_1d(s["low_action"]) for s in episode_state
            ])
            grp.create_dataset("low_action", data=low_action_data, compression="gzip")
        # ==== 分类统计 ====
        if end_state == "success":
            success_times += 1
            count_time.append(env.step_counter * env.dt)
        elif end_state == "collide":
            collide_times += 1
        elif end_state == "outbound":
            outbound_times += 1
        elif end_state == "timelimit":
            timeout_times += 1

        # ==== 保存 CSV ====
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
        results.clear()

    env.close()
    if _save_state:
        state_h5_file.close()
        print(colored(f"Saved states to {state_h5_path}", 'cyan'))

    average_time = np.mean(count_time) if count_time else float('nan')
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print(f"State_dim: {state_dim_mode} || Average Navigation Time: {average_time}")
    print(colored(f"Saved test results to {csv_path}", 'cyan'))
    if _save_traj:
        print(colored(f"Saved trajectories to {traj_pickle_path}", 'cyan'))
###########################

########################### Mode 2: RL_test ###########################
def success_rate_test_RL(_id_in=1):
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
    video_dir = "./Video_RL/video_frames"
    #####################################################  

    ##################################################### Tasks #####################################################  
    ########### With Obstacle ###########
    # _save_traj = False
    # _extra_obs = True
    # random_range = 32
    
    # _save_state = True
    # _save_traj = False
    # _extra_obs = False
    # random_range = 56
        
    # _id = 3
    # if _save_traj:
    #     # test_mode = True
    #     test_mode = False
    #     switch_range = 8
    #     flow_seed = 1
    # else:
    #     test_mode = False
    #     switch_range = 8
    #     flow_seed = 0 
    # switch_mode = False
    # _is_normalize = True
    # _fov = np.pi
    # _start = np.array([220.0, 64.0])
    # _target = np.array([64.0, 64.0])
    # _start_position = None
    # _target_position = None
    # max_steps = 600
    # # max_steps = 750
    # max_ep_len = max_steps + 20
    # state_dim_mode = 10
    # # state_dim_mode = 8
    # input_max = 15.0
    # clip_max = 15.0  
    # ppo_path = f'./models/exp/rl/dim{state_dim_mode}/{_id}.pth'

    ########### No obstacle ###########
    _save_state = True
    # _save_state = False
    _save_traj = True
    # _save_traj = False
    _extra_obs = False
    if _save_traj:
        flow_seed = 15
        # test_mode = True
        # switch_range = 8
        test_mode = False
        switch_range = 4
    else:
        flow_seed = 0
        test_mode = False
        switch_range = 8
    switch_mode = False
    _is_normalize = True
    _fov = np.pi
    _start = np.array([240.0, 64.0])
    _target = np.array([160.0, 64.0])
    _start_position = None
    _target_position = None
    random_range = 32
    max_steps = 200
    max_ep_len = max_steps + 20
    input_max = 15.0
    clip_max = 15.0
    _id = 9086
    # state_dim_mode = 11 # 9070 - 9074
    # state_dim_mode = 6 # 9080 - 9084 best:9083
    state_dim_mode = 3 # 9080 - 9082 9085 9086
    ppo_path = f'./models/exp/rl/dim{state_dim_mode}/{_id}.pth'
 
    ######################################################
    env = train_env_basic_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
    _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, _is_switch=switch_mode, u_clip=clip_max , v_clip=clip_max, _fov=_fov)
    ######################################################

    ########################### State & Action ###########################
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
    # initialize RL agent
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)

    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=20)
    # ppo_agent.load_full_icm(ppo_path)
    ######################################################

    ########################### Test ###########################
    if _save_traj:
        test_time = 20
    else:
        test_time = 2000
    success_times = 0
    collide_times = 0
    timeout_times = 0
    outbound_times = 0
    save_interval = 100
    count_time = []

    # === CSV 文件初始化（保留原来的 episode 统计） ===
    csv_path = f"./Experiment/RL_a15_test_seed_{random_seed}_dim_{state_dim_mode}_model_{_id}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results = []

    if _save_traj:
        traj_pickle_path = f"./Experiment/RL_trajectories_seed_{random_seed}_dim_{state_dim_mode}_model_{_id}.pkl"
        os.makedirs(os.path.dirname(traj_pickle_path), exist_ok=True)

    # 首次写入 CSV 表头
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Start_X", "Start_Y", "Target_X", "Target_Y", "Flow_ID", "End_State", "End_angle", "End_time"])

    # === HDF5 文件初始化，用于 _save_state ===
    if _save_state:
        state_h5_path = f"./Experiment/RL_states_seed_{random_seed}_dim_{state_dim_mode}_model_{_id}.h5"
        os.makedirs(os.path.dirname(state_h5_path), exist_ok=True)
        state_h5_file = h5py.File(state_h5_path, "a")

    # Test
    for n in range(test_time):
        # 1 env reset
        # state, info = env.reset()
        state, info = env.reset(_add_virtual_cylinders=_extra_obs, _custom_cylinders=[(160, 64), (160, 108), (160, 16)])
        start_position = env.agent_pos.copy()
        print(f"Episode{n} Start")
        print("position_reset:", env.agent_pos)
        flow_id = env.flow_init

        if _save_traj:
            # 保存当前集完整轨迹
            episode_trajectory = []  

        if _save_state:
            # 保存每步 state/action/t
            episode_state = []

        for i in range(1, max_ep_len + 1):
            # 2 env step
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            agent_position = env.agent_pos
            done = terminated or truncated
            end_state = env.end_state

            # ==== 保存轨迹数据 ====
            if _save_traj:
                episode_trajectory.append({
                    "step": i,
                    "agent_x": agent_position[0].copy(),
                    "agent_y": agent_position[1].copy(),
                    "done": done
                })

            # ==== 保存 state/action/t 到 HDF5 ====
            if _save_state:
                episode_state.append({
                    "t": i,
                    "state": state.copy(),
                    "action": action.tolist() if hasattr(action, 'tolist') else action
                })

            if done:      
                if end_state == "success":
                    print(colored(f"Episode{n} success", 'green'))
                    success_times += 1
                    count_time.append(env.step_counter * env.dt)
                    break
                elif end_state == "collide":
                    print(colored(f"Episode{n} collide", 'yellow'))
                    collide_times += 1
                    break
                elif end_state == "outbound":
                    print(colored(f"Episode{n} outbound", 'red'))
                    outbound_times += 1
                    break
                elif end_state == "timelimit":
                    print(colored(f"Episode{n} timeout", 'blue'))
                    timeout_times += 1
                    break

        # ==== Episode 结束 ====
        end_angle = env.angle
        start_x, start_y = start_position
        target_x, target_y = env.target_position
        results.append([n, start_x, start_y, target_x, target_y, flow_id, end_state, end_angle, i])

        # ==== 保存轨迹到 Pickle ====
        if _save_traj:
            with open(traj_pickle_path, "ab") as f:
                pickle.dump(episode_trajectory, f)

        # ==== 保存 HDF5 state/action/t ====
        if _save_state and len(episode_state) > 0:
            grp_name = f"episode_{n:05d}"
            if grp_name in state_h5_file:
                del state_h5_file[grp_name]
            grp = state_h5_file.create_group(grp_name)
            grp.create_dataset("t", data=np.array([s["t"] for s in episode_state]), compression="gzip")
            grp.create_dataset("state", data=np.array([s["state"] for s in episode_state]), compression="gzip")
            action_data = np.array([s["action"] for s in episode_state])
            grp.create_dataset("action", data=action_data, compression="gzip")

        # ==== 每隔 save_interval 保存一次 CSV ====
        if (n + 1) % save_interval == 0 or n == test_time - 1:
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)
            results.clear()

    # ==== 关闭环境和 HDF5 文件 ====
    env.close()
    if _save_state:
        state_h5_file.close()
        print(colored(f"Saved states to {state_h5_path}", 'cyan'))

    average_time = np.mean(count_time)
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print(f"State_dim: {state_dim_mode} || Average Navigation Time: {average_time}")
###########################

def test_1(_id_in=1):
    ########################### Mode 3: RL Single test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 300
    max_ep_len = max_steps + 20
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    state_dim_mode = 6
    ppo_path = f'./models/exp/rl/dim6/9083.pth'

    # state_dim_mode = 11
    # ppo_path = f'./models/exp/rl/dim11/9071.pth'


    # state_dim_mode = 3
    # ppo_path = f'./models/exp/rl/dim3/9080.pth'


    # set flow seed if required (0 : random flow; else : fixed flow)
    # flow_seed = 0
    flow_seed = 10

    # set random mode(True : real random mode; else : random.seed(random_seed))
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _id_in
    
    test_mode = False
    switch_range = 4
    
    # switch_mode = True
    switch_mode = False

    _is_normalize = True
    # _is_normalize = False

    input_max = 15.0
    clip_max = 15.0  

    ######################################################
    _start = np.array([float(240), float(64)])
    _target = np.array([float(160), float(64)])
    random_range = 32
    env = train_env_basic_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize,
    _include_flow=False, _plot_flow=False, _proccess_flow=False,  
    _is_test = test_mode, _state_dim=state_dim_mode, 
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
    # ppo_agent_1 = Classic_PPO(14, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias)
    # ppo_agent_1.load_full(ppo_path_1)
    # ppo_agent_1.set_eval_mode(True)
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=20)
    # ppo_agent.load_full_icm(ppo_path)
    ######################################################

    ########################### Test ###########################
    # 设置保存目录
    save_dir = f"./10_rl_traj/dim_{state_dim_mode}_test_results"
    os.makedirs(save_dir, exist_ok=True)

    num_tests = 20  # 测试次数
    max_ep_len = 300  # 每次测试最大步数

    all_test_data = []

    for test_id in range(1, num_tests + 1):
        # ---------------- Reset environment ----------------
        state, info = env.reset()
        print(f"Test {test_id} reset: Start Pos: {env.agent_pos}, Target Pos: {env.target_position}")

        ep_return = 0
        total_steps = 0
        result = None  # 用于记录成功/失败

        # 记录信息
        trajectory = [env.agent_pos.copy()]
        angles = [env.angle]
        action_ppo = []
        pressure_ppo = []
        speed_ppo = []

        for step in range(1, max_ep_len + 1):
            # ---------------- Select action ----------------
            action = ppo_agent.select_action(state)
            action_ppo.append(action)

            # ---------------- Step environment ----------------
            state, reward, terminated, truncated, info = env.step(action)

            # ---------------- Record ----------------
            ep_return += reward
            total_steps = step
            trajectory.append(env.agent_pos.copy())
            angles.append(env.angle)
            pressure_ppo.append([info[f"pressure_{i}"] for i in range(1, 9)])
            speed = math.sqrt(info["vel_x"]**2 + info["vel_y"]**2)
            speed_ppo.append([info["vel_x"], info["vel_y"], info["vel_angle"], speed])

            if terminated or truncated:
                result = "Success" if terminated else "Failure"
                break

            # ---------------- Clear buffer ----------------
            ppo_agent.buffer.clear()

        if result is None:
            result = "Failure"  # 如果未提前结束，则视为失败

        env.close()
        print(f"Test {test_id} finished: Total Reward={ep_return}, Total Steps={total_steps}, Result={result}")

        # ---------------- Save per-test data ----------------
        test_data = {
            "test_id": test_id,
            "start_pos": env.start_position,
            "target_pos": env.target_position,
            "trajectory": np.array(trajectory),
            "angles": np.array(angles),
            "actions": np.array(action_ppo),
            "pressures": np.array(pressure_ppo),
            "speeds": np.array(speed_ppo),
            "total_reward": ep_return,
            "total_steps": total_steps,
            "result": result
        }
        all_test_data.append(test_data)

    # ---------- Save all test data to CSVs ----------
    for test_data in all_test_data:
        tid = test_data["test_id"]

        # 1. Trajectory + angles + result
        traj_file = os.path.join(save_dir, f"trajectory_test{tid}.csv")
        with open(traj_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "angle", "result"])
            for (pos, angle) in zip(test_data["trajectory"], test_data["angles"]):
                writer.writerow([pos[0], pos[1], angle, test_data["result"]])

        # 2. Actions
        action_file = os.path.join(save_dir, f"action_test{tid}.csv")
        with open(action_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["a_x", "a_y", "a_w"])
            for a in test_data["actions"]:
                writer.writerow(a)

        # 3. Pressures
        pressure_file = os.path.join(save_dir, f"pressure_test{tid}.csv")
        with open(pressure_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"p_{i}" for i in range(1, 9)])
            for p in test_data["pressures"]:
                writer.writerow(p)

        # 4. Speeds
        speed_file = os.path.join(save_dir, f"speed_test{tid}.csv")
        with open(speed_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["v_x", "v_y", "v_angle", "v"])
            for s in test_data["speeds"]:
                writer.writerow(s)

        # 5. Summary info
        summary_file = os.path.join(save_dir, f"summary_test{tid}.txt")
        with open(summary_file, "w") as f:
            f.write(f"Test ID: {tid}\n")
            f.write(f"Start Position: {test_data['start_pos']}\n")
            f.write(f"Target Position: {test_data['target_pos']}\n")
            f.write(f"Total Steps: {test_data['total_steps']}\n")
            f.write(f"Total Reward: {test_data['total_reward']}\n")
            f.write(f"Result: {test_data['result']}\n")

    print(f"All {num_tests} tests saved to {save_dir}")

########################### Mode 4: Hovering Test Controller###########################
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
    force_range = 15.0
    force_clip = 15.0
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
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
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


def Hovering_test_RL(_id_in=1):
    ########################### Mode 1: Multi-Test ###########################
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
    force_range = 15.0
    force_clip = 15.0
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
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
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


def Hovering_test_HRL_2(_id_in=1):
    ########################### Mode 1: Multi-Test ###########################
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
    force_range = 15.0
    force_clip = 15.0
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
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
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


def PathTracking_test_HRL(_id_in=1):
    ########################### Mode 7: Path Tracking ###########################
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
    video_dir = "./Video_10_Tracking/video_frames"
    max_ll_steps = 5
    #####################################################  

    ##################################################### Tasks #####################################################  

    ########### 低速小范围正常任务 ###########
    test_mode = False
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
    # max_steps = 1500
    max_steps = 250
    max_ep_len = max_steps + 20
    # max_ep_len = 200
    state_dim_mode = 10
    lc_state_dim = 6
    force_range = 15.0
    force_clip = 15.0
    # ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
    _is_hrl = True
    if _is_hrl:
        ppo_path_1 = f'./models/exp/lc_dim6.pth'
    else:
        ppo_path_1 = f'./models/exp/rl/dim6/9083.pth'
    ##################################################### ///// #####################################################

    ########### Environment ###########
    flow_seed = 0  
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
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
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

    # ########################### Trajectory Generation ###########################
    # n_points = 250
    # n_points = 500
    n_points = 50

    margin_1 = 20
    margin_2 = 10

    # 边界
    x_min, x_max = 125, 300
    y_min, y_max = 15, 120

    trajectory_list = []
    trajectory_names = []

    # -------------------- RL vs. HRL --------------------
    R = 24
    fixed_start_points = [(180, 64), (200, 40)]  # center

    for center_x, center_y in fixed_start_points:
        t = np.linspace(0, np.pi, n_points)   # half circle
        x = center_x + R * np.cos(t)          # +R to -R
        y = center_y + R * np.sin(t)

        arc_points = np.vstack((x, y)).T
        trajectory_list.append(arc_points)

        if _is_hrl:
            trajectory_names.append(f"semicircle_R{R}_center_{center_x}_{center_y}")
        else:
            trajectory_names.append(f"rl_semicircle_R{R}_center_{center_x}_{center_y}")


    # # -------------------- 无限形轨迹生成 --------------------
    # # 参数组 1: a > b
    # a1, b1 = 50, 20
    # start_x_range1 = [x_min + margin_1 + a1, x_max - margin_1 - a1]
    # start_y_range1 = [y_min + margin_1 + b1, y_max - margin_1 - b1]

    # for start_x, start_y in [(start_x_range1[0], start_y_range1[0]),
    #                         (start_x_range1[1], start_y_range1[1])]:
    #     # t从0到2pi，起点在中心
    #     t = np.linspace(0, 2*np.pi, n_points)
    #     x = a1 * np.sin(t) + start_x  # 平移到中心点
    #     y = b1 * np.sin(2*t) + start_y
    #     arc_points = np.vstack((x, y)).T
    #     trajectory_list.append(arc_points)
    #     trajectory_names.append(f"figure8_a{a1}_b{b1}_start_{start_x}_{start_y}")


    # 参数组 2: b > a
    # a2, b2 = 20, 40
    # 起点固定
    # fixed_start_points = [(200, 64), (240, 64)]
    # a2, b2 = 50, 20
    # fixed_start_points = [(195, 55), (230, 80)]

    # for start_x, start_y in fixed_start_points:
    #     t = np.linspace(0, 2*np.pi, n_points)
    #     x = a2 * np.sin(t) + start_x
    #     y = b2 * np.sin(2*t) + start_y
    #     arc_points = np.vstack((x, y)).T
    #     trajectory_list.append(arc_points)
    #     if _is_hrl:
    #         trajectory_names.append(f"figure8_a{a2}_b{b2}_start_{start_x}_{start_y}")
    #     else:
    #         trajectory_names.append(f"rl_figure8_a{a2}_b{b2}_start_{start_x}_{start_y}")


    # # ---------------- 圆形曲线（起点左侧） ----------------
    # r_circle = 30
    # start_x_circle = x_min + margin_1 + r_circle
    # start_y_circle = y_min + margin_1 + r_circle
    # t = np.linspace(0, 2*np.pi, n_points)
    # x = r_circle * np.cos(t + np.pi) + start_x_circle
    # y = r_circle * np.sin(t + np.pi) + start_y_circle
    # trajectory_list.append(np.vstack((x, y)).T)
    # trajectory_names.append(f"circle_r{r_circle}_start_{start_x_circle}_{start_y_circle}")

    # # ---------------- 椭圆曲线（起点左侧） ----------------
    # a_ellipse, b_ellipse = 40, 20
    # start_x_ellipse = x_min + margin_1 + a_ellipse
    # start_y_ellipse = y_min + margin_1 + b_ellipse
    # t = np.linspace(0, 2*np.pi, n_points)
    # x = a_ellipse * np.cos(t + np.pi) + start_x_ellipse
    # y = b_ellipse * np.sin(t + np.pi) + start_y_ellipse
    # trajectory_list.append(np.vstack((x, y)).T)
    # trajectory_names.append(f"ellipse_a{a_ellipse}_b{b_ellipse}_start_{start_x_ellipse}_{start_y_ellipse}")

    # ---------------- Tracking Execution ----------------
    save_dir = f"./Experiment_tracking_test_results_points_{n_points}"
    os.makedirs(save_dir, exist_ok=True)

    flow_list = range(1, 25)  # 流场编号
    near_threshold = env.max_detect_dis + env.window_r/2
    for traj_points, traj_name in zip(trajectory_list, trajectory_names):
        print(colored(f"\n===== Testing Trajectory: {traj_name} =====", "cyan"))

        for flow_id in flow_list:
            print(colored(f"--- Flow Field {flow_id} ---", "green"))
            results = []

            # 初始化环境，第一个目标点
            state, info = env.reset(
                _flow_init=flow_id,
                _start_position_init=traj_points[0],
                _target_position_init=traj_points[1],
                _verbose=False
            )
            state_14 = env.state_14
            agent_position = env.agent_pos.copy()

            i = 1
            t_step = 0
            done = False

            while not done and t_step < max_ep_len:
                target_position = env.target_position
                lc_target_x, lc_target_y = target_position
                env.set_lc_target([lc_target_x, lc_target_y])

                # 构造低层状态
                low_state = state_14.copy()
                if _is_hrl:
                    low_state[3] = lc_target_x - agent_position[0]
                    low_state[4] = lc_target_y - agent_position[1]
                else:
                    low_state[3] = (lc_target_x - agent_position[0])/near_threshold
                    low_state[4] = (lc_target_y - agent_position[1])/near_threshold
                if lc_state_dim == 3:
                    low_state = low_state[3:6]
                elif lc_state_dim == 6:
                    low_state = low_state[:6]
                elif lc_state_dim == 11:
                    low_state = np.hstack([low_state[3:6], state_14[6:14]]).astype(np.float32)

                # 低层动作
                low_action = ppo_agent_lc.select_action(low_state)
                cliped_low_action = env.clip_action(low_action)

                # 环境执行
                state, reward, terminated, truncated, info = env.step(low_action)
                agent_position = env.agent_pos.copy()
                state_14 = env.state_14.copy()

                # 记录每步
                results.append({
                    "flow_id": flow_id,
                    "step": t_step,
                    "position": agent_position.copy(),
                    "speed": env.speed,
                    "velocity": env.agent_velocity.copy(),
                    "agent_angle": env.angle,
                    "omega": env.vel_angle,
                    "action": cliped_low_action.copy(),
                    "target_position": [lc_target_x, lc_target_y],
                    "trajectory_name": traj_name
                })

                t_step += 1
                if terminated:
                    i += 1
                    if i < len(traj_points):
                        env.set_target(traj_points[i])
                    else:
                        done = True
                        print(colored(f"All targets reached at step {t_step}", "green"))
                if truncated:
                    print(colored(f"Truncated at step {t_step}", "red"))
                    break

            # 保存结果，每条轨迹每个流场独立文件
            save_path = os.path.join(save_dir, f"{traj_name}_flow_{flow_id}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print(colored(f"Saved: {save_path}", "green"))
            results.clear()

    env.close()
    print(colored("\nAll trajectories and flows tested and saved.", "blue")) 

    # ########################### Tracking Path ###########################
    # a = 50   # 横向振幅（宽度的一半）
    # b = 20   # 纵向振幅（高度的一半）
    # n_points = 500  # 轨迹采样点数

    # # 起点目标
    # start_x = 180
    # start_y = 72

    # # 找一个起点相位（这里我们直接设 t=0 对应起点）
    # t = np.linspace(0, 2*np.pi, n_points)

    # # 原始曲线（相对于 0,0）
    # x = a * np.sin(t)
    # y = b * np.sin(2*t)

    # # 平移到 (180, 72)
    # x += start_x
    # y += start_y

    # points = np.vstack((x, y)).T
    # arc_points = points
    # ###################################################### 

    # ########################### Test ###########################
    # save_dir = "./Experiment_tracking_test_results"
    # os.makedirs(save_dir, exist_ok=True)

    # flow_list = range(1, 25)
    # subgoal_threshold = 1

    # for flow_id in flow_list:
    #     print(colored(f"\n===== Testing Flow Field {flow_id} =====", "green"))

    #     results = []

    #     state, info = env.reset(
    #         _flow_init=flow_id,
    #         _start_position_init=arc_points[0],
    #         _target_position_init=arc_points[1],
    #         _verbose=False
    #     )
    #     state_14 = env.state_14
    #     agent_position = env.agent_pos.copy()
    #     i = 1
    #     t = 0
    #     done = False

    #     while not done and t < max_ep_len:
    #         target_position = env.target_position
    #         lc_target_x, lc_target_y = target_position
    #         env.set_lc_target([lc_target_x, lc_target_y])

    #         low_state = state_14.copy()
    #         low_state[3] = lc_target_x - agent_position[0]
    #         low_state[4] = lc_target_y - agent_position[1]
    #         if lc_state_dim == 3:
    #             low_state = low_state[3:6]
    #         elif lc_state_dim == 6:
    #             low_state = low_state[:6]
    #         elif lc_state_dim == 11:
    #             low_state = np.hstack([low_state[3:6], state_14[6:14]]).astype(np.float32)

    #         low_action = ppo_agent_lc.select_action(low_state)
    #         cliped_low_action = env.clip_action(low_action)

    #         state, reward, terminated, truncated, info = env.step(low_action)
    #         agent_position = env.agent_pos.copy()
    #         state_14 = env.state_14.copy()
    #         # t时刻的动作和目标 t+1时刻的状态
    #         results.append({
    #             "flow_id": flow_id,
    #             "step": t,
    #             "position": agent_position.copy(),
    #             "speed": env.speed,
    #             "velocity": env.agent_velocity.copy(),
    #             "agent_angle": env.angle,
    #             "omega": env.vel_angle,
    #             "action": cliped_low_action.copy(),
    #             "target_position": [lc_target_x, lc_target_y]
    #         })

    #         t += 1
    #         if terminated:
    #             i += 1
    #             if i < len(arc_points):
    #                 env.set_target(arc_points[i])
    #             else:
    #                 done = True
    #                 print(colored(f"All targets reached at step {t}", "green"))
    #         if truncated:
    #             print(colored(f"Truncated at step {t}", "red"))
    #             break

    #     # 每个流场测试完保存一次
    #     save_path = os.path.join(save_dir, f"controller_tracking_flow_{flow_id}.pkl")
    #     with open(save_path, "wb") as f:
    #         pickle.dump(results, f)
    #     print(colored(f"Flow {flow_id} data saved: {save_path}", "green"))

    #     # 清空结果，释放内存
    #     results.clear()

    # env.close()
    # print(colored("\nAll flows tested and saved individually.", "blue"))
    # ########################### \\\\\\\\\\\\ ###########################

    # ########################### Test ###########################
    # print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    # env.set_traj(arc_points)

    # subgoal_threshold = 1
    # flow_init = 1
    # i = 1
    # state, info = env.reset(_flow_init=None, _start_position_init=arc_points[0], _target_position_init=arc_points[1], _verbose=False)

    # # Store initial states
    # state_14 = env.state_14
    # agent_position = env.agent_pos
    # print("position_reset:", agent_position)
    

    # # Main episode loop
    # t = 0
    # hl_time_step = 0
    # done = False

    # while not done and t < max_ep_len:
    #     target_position = env.target_position
    #     lc_target_x, lc_target_y = target_position
    #     # === Normal Subgoal ===
    #     env.set_lc_target([lc_target_x, lc_target_y])
    #     print(f"=== Subgoal: ({lc_target_x}, {lc_target_y}) ===")
    #     # Low-level state update
    #     low_state = state_14.copy()
    #     low_state[3] = lc_target_x - agent_position[0]
    #     low_state[4] = lc_target_y - agent_position[1]
    #     if lc_state_dim == 3:
    #         low_state = low_state[3:6]
    #     elif lc_state_dim == 6:
    #         low_state = low_state[:6]
    #     elif lc_state_dim == 11:
    #         low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)
    #     # Low-level action
    #     low_action = ppo_agent_lc.select_action(low_state)
    #     cliped_low_action = env.clip_action(low_action)
    #     print(f"=== Low Level Step {t}, Action: {low_action}")
    #     print(f"===               -> Clip_Action: {cliped_low_action} ===")
    #     # Step env
    #     state, reward, terminated, truncated, info = env.step(low_action)
    #     agent_position = env.agent_pos
    #     state_14 = env.state_14
    #     t += 1
    #     if terminated:
    #         i += 1
    #         env.set_target(arc_points[i])
    #     if truncated:
    #         break
    # env.close()
    # print("Total_Steps:", t)


def test_fixed(_id_in=1):
    ########################### Mode 8: Fixed Target Test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 200
    max_ep_len = max_steps + 20
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    # state_dim_mode = 3
    # ppo_path = f'./models/exp/rl/dim3/9080.pth'

    # state_dim_mode = 11
    # ppo_path = f'./models/exp/rl/dim11/9071.pth'

    state_dim_mode = 6
    # ppo_path = f'./models/exp/rl/dim6/9083.pth'
    ppo_path = f'./models/exp/lc_dim6.pth'

    # set flow seed if required (0 : random flow; else : fixed flow)
    # flow_seed = 0
    flow_seed = 25

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

    # _is_normalize = True
    _is_normalize = False

    input_max = 15.0
    clip_max = 15.0  

    ######################################################
    _start = np.array([float(160), float(96)])
    _target = np.array([float(160), float(96)])
    random_range = 32
    env = train_env_basic_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start, target_position=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize,
    _include_flow=True, _plot_flow=True, _proccess_flow=False,  
    _is_test = test_mode, _state_dim=state_dim_mode, video_dir="./Experiment_Hovering/video_frames",
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
        # 3 save info
        ep_return_ppo += reward
        # if terminated or truncated:
        #     break
    env.close()
    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)
    ######################################################


########################### Mode 9: HRL Single Test Heading with Obstacle ###########################
def test_heading(_id_in=1):

    print("============================================================================================")

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    ################ RL Hyperparameters ################
    has_continuous_action_space = True
    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0001
    lr_critic = 0.0002
    action_std_4_test = 0.04

    ################ Env / Task Settings ################
    true_random = True
    random_seed = 0 if true_random else _id_in
    _is_head = False
    max_ll_steps = 5
    flow_seed = _id_in
    test_mode = True
    switch_range = 8
    theta_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    switch_mode = False
    force_range = 15.0
    force_clip = 15.0
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    random_range = 56
    max_steps = 600
    max_ep_len = max_steps + 20
    state_dim_mode = 10
    lc_state_dim = 6
    _extra_obs = True

    # PPO paths
    ppo_path_lc = f'./models/exp/lc_dim6.pth'

    video_dir = "./12_Video_Obstacle/video_frames"

    ########################### Env ###########################
    # env = train_env_upper_2.foil_env(
    #     args_1, max_step=max_steps, start_center=_start, target_center=_target,
    #     start_position=None, target_position=None,
    #     _include_flow=True, _plot_flow=True, _proccess_flow=False,
    #     _random_range=random_range, _init_flow_num=flow_seed,
    #     _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode,
    #     _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
    #     u_range=force_range, v_range=force_range, u_clip=force_clip, v_clip=force_clip,
    #     _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode=theta_mode,
    #     _is_switch=switch_mode, _test_info=False, video_dir=video_dir,
    #     _near_mode=_is_near, _fov=_fov
    # )

    env = train_env_plus.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=None, target_position=None,
        _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov
    )

    ########################### Action Space ###########################
    state_dim = env.observation_space.shape[0]
    lc_action_dim = env.lc_action_space.shape[0]
    lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
    lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0

    ########################### Low-level PPO ###########################
    ppo_agent_lc = Classic_PPO(
        lc_state_dim, lc_action_dim,
        lr_actor, lr_critic, gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std_init=action_std_4_test,
        continuous_action_output_scale=lc_action_output_scale,
        continuous_action_output_bias=lc_action_output_bias
    )
    ppo_agent_lc.load_full(ppo_path_lc)
    ppo_agent_lc.set_eval_mode(True)

    ########################### Test Initialization ###########################
    print(colored("**** Geometric Heading Mode with Obstacle Avoidance ****", 'green'))

    # _start_position_init = np.array([180, 96])
    # _target_position_init = np.array([240, 64])
    _start_position_init = np.array([220, 32])
    _target_position_init = np.array([64, 64])

    # state, info = env.reset(_start_position_init=_start_position_init,
    #                         _target_position_init=_target_position_init)

    state, info = env.reset(_start_position_init=_start_position_init, _target_position_init=_target_position_init, _add_virtual_cylinders=_extra_obs, _custom_cylinders=[(180, 64), (180, 108), (180, 16)])

    state_14 = env.state_14
    agent_position = env.agent_pos
    target_position = env.target_position

    print("position_reset:", agent_position)

    # Record containers
    record_high_states = []
    record_positions = []

    ########################### Main Loop ###########################
    t = 0
    done = False

    while not done and t < max_ep_len:

        # Record states & positions
        record_high_states.append(state.copy())
        record_positions.append(agent_position.copy())

        # ================= Heading Calculation =================
        vec_to_target = target_position - agent_position
        dist_to_target = np.linalg.norm(vec_to_target)
        if dist_to_target < 1e-6:
            break

        # Base angle towards target
        angle_to_use = np.arctan2(vec_to_target[1], vec_to_target[0])
        r = min(2.0, dist_to_target)

        # Extract nearest obstacle info from state (state_dim=10)
        barricade_state = state[6:8]
        direction = state[8]

        # If obstacle detected, adjust subgoal along tangent using direction
        if abs(direction) > 0:
            dx, dy = barricade_state

            if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
                # fallback: small step along current heading
                lc_target_x = agent_position[0] + r * np.cos(angle_to_use)
                lc_target_y = agent_position[1] + r * np.sin(angle_to_use)
            else:
                # Tangent vector: rotate (dx, dy) by 90° * direction (corrected)
                tangent_vec = np.array([dy, -dx]) * np.sign(direction)  # 修正旋转方向
                tangent_vec /= np.linalg.norm(tangent_vec) + 1e-8       # normalize
                lc_target_x = agent_position[0] + r * tangent_vec[0]
                lc_target_y = agent_position[1] + r * tangent_vec[1]
        else:
            # Normal heading towards target
            lc_target_x = agent_position[0] + r * np.cos(angle_to_use)
            lc_target_y = agent_position[1] + r * np.sin(angle_to_use)

        # Set low-level subgoal
        env.set_lc_target([lc_target_x, lc_target_y])

        # ================= Low-level Control =================
        low_state = state_14.copy()
        low_state[3] = lc_target_x - agent_position[0]
        low_state[4] = lc_target_y - agent_position[1]

        if lc_state_dim == 3:
            low_state = low_state[3:6]
        elif lc_state_dim == 6:
            low_state = low_state[:6]
        elif lc_state_dim == 11:
            low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

        low_action = ppo_agent_lc.select_action(low_state)
        low_action = env.clip_action(low_action)

        state, reward, terminated, truncated, info = env.step(low_action)

        done = terminated or truncated
        agent_position = env.agent_pos
        state_14 = env.state_14

        t += 1

    ########################### End ###########################
    env.close()
    print("Total_Steps:", t)

    ########################### Save ###########################
    save_dir = "./Experiment/Single_test_obstacle/"
    os.makedirs(save_dir, exist_ok=True)

    record_high_states = np.array(record_high_states)
    record_positions = np.array(record_positions)

    np.save(os.path.join(save_dir, f"record_high_states_dim_{state_dim_mode}.npy"), record_high_states)
    np.save(os.path.join(save_dir, f"record_positions_dim_{state_dim_mode}.npy"), record_positions)

    print(f"Saved: {save_dir}record_high_states_dim_{state_dim_mode}.npy")
    print(f"Saved: {save_dir}record_positions_dim_{state_dim_mode}.npy")


########################### Mode 10: Multi-Test Heading ###########################
def success_rate_test_Heading(_id_in=1):
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
    np.random.seed(random_seed)
    #####################################################  

    ##################################################### Tasks ##################################################### 
    _save_traj = True
    # _save_traj = False
    # _save_state = True
    _save_state = False 

    ########### 避障任务 ###########
    # _extra_obs = True
    _extra_obs = False

    if _save_traj:
        test_mode = True
        switch_range = 8
        flow_seed = 1
    else:
        # test_mode = True
        test_mode = False
        switch_range = 8
        flow_seed = 0
    theta_mode = False
    switch_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    _start_position = None
    _target_position = None

    # random_range = 56
    random_range = 32

    # max_steps = 400
    # max_steps = 750
    max_steps = 600

    max_ep_len = max_steps + 20

    state_dim_mode = 10
    model_id = 1
    

    # state_dim_mode = 8
    # model_id = 5

    lc_state_dim = 6
    force_range = 15.0
    force_clip = 15.0
    ppo_path = f'./models/exp/hrl/dim{state_dim_mode}/a15+fov180_{model_id}.pth'

    # rl_control = True
    # _id = 9083
    # ppo_path_1 = f'./models/exp/rl/dim{lc_state_dim}/{_id}.pth'

    rl_control = False
    ppo_path_1 = f'./models/exp/lc_dim6.pth'


    ########### 高速精细避障任务 ###########
    # flow_seed = 0 
    # test_mode = False
    # switch_range = 4
    # theta_mode = False
    # switch_mode = False
    # _is_normalize = True
    # _is_avoid = True
    # _is_near = True
    # _fov = 1.5 * np.pi
    # _start = np.array([220.0, 64.0])
    # _target = np.array([64.0, 64.0])
    # random_range = 32
    # max_steps = 240
    # max_ep_len = max_steps + 20
    # state_dim_mode = 10
    # lc_state_dim = 14
    # force_range = 20.0
    # force_clip = 20.0
    # ppo_path = f'./models/exp/hrl/dim10/a20+fov270+near.pth'
    # ppo_path_1 = f'./models/exp/lc_dim14_a20.pth'

    
    ########### 无障碍任务 ###########
    # flow_seed = 0 
    # test_mode = False
    # switch_range = 8
    # theta_mode = False
    # switch_mode = False
    # _is_normalize = True
    # _is_avoid = False
    # _is_near = False
    # _start = np.array([240.0, 64.0])
    # _target = np.array([160.0, 64.0])
    # random_range = 32
    # max_steps = 200
    # max_ep_len = max_steps + 20

    ##################################################### ///// ##################################################### 

    
    ########### Environment ########### 
    # env = train_env_upper_2.foil_env(
    #     args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
    #     _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, 
    #     _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
    #     _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
    #     u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
    #     _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
    #     _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov = _fov
    # )

    env = train_env_plus.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
        _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov
    )

    ################ Action Space Settings ################
    state_dim = env.observation_space.shape[0]
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
    subgoal_threshold = 1
    test_time = 20 if _save_traj else 2000
    success_times = 0
    collide_times = 0
    timeout_times = 0
    outbound_times = 0
    count_time = []

    # print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    print(colored(f"State dimension: {state_dim_mode} | Model ID: {model_id}", 'green'))
    save_items = []
    if _save_traj:
        save_items.append("trajectory (pickle)")
    if _save_state:
        save_items.append("state (HDF5)")

    if save_items:
        print(colored(f"Will save: {', '.join(save_items)}", 'yellow'))
    else:
        print(colored("No data will be saved.", 'yellow'))
    
    # === CSV 文件初始化（保留原来的 episode 统计） ===
    if rl_control:
        csv_path = f"./Experiment/Heading_a15_test_seed_{random_seed}_dim_{state_dim_mode}_model_RL.csv"
    else:
        csv_path = f"./Experiment/Heading_a15_test_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Start_X", "Start_Y", "Target_X", "Target_Y", "Flow_ID", "End_State", "End_angle", "End_time"])

    # === Trajectory pickle 初始化 ===
    if _save_traj:
        traj_pickle_path = f"./Experiment/Heading_trajectories_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}.pkl"
        os.makedirs(os.path.dirname(traj_pickle_path), exist_ok=True)

    # === HDF5 文件初始化，用于 _save_state ===
    if _save_state:
        state_h5_path = f"./Experiment/Heading_states_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}.h5"
        os.makedirs(os.path.dirname(state_h5_path), exist_ok=True)
        state_h5_file = h5py.File(state_h5_path, "a")
    

    # ==== 主测试循环 ====
    for n in range(test_time):
        # ==== Env Reset ====
        # center 200
        # x_start = 260.0
        # x_target = 140.0
        # y_start = np.random.uniform(32, 96)
        # y_target = np.random.uniform(32, 96)
        # _start_position_init = np.array([x_start, y_start])
        # _target_position_init = np.array([x_target, y_target])
        # state, info = env.reset(_start_position_init=_start_position_init, _target_position_init=_target_position_init, _add_virtual_cylinders=_extra_obs, _custom_cylinders=[(200, 64), (200, 108), (200, 16)])
        # ==== Env Reset ====
        state, info = env.reset(_add_virtual_cylinders=_extra_obs, 
                                _custom_cylinders=[(160, 64), (160, 108), (160, 16)])
        state_14 = env.state_14
        agent_position = env.agent_pos
        start_position = np.copy(agent_position)
        target_position = np.copy(env.target_position)
        flow_id = env.flow_init

        episode_traj = []    # traj saving
        episode_state = []   # state saving
        results = []

        t = 0
        done = False

        # ================= Main Loop =================
        while not done and t < max_ep_len:

            agent_position = env.agent_pos
            agent_x, agent_y = agent_position
            target_x, target_y = env.target_position

            # ==== Record states & positions ====
            # record_high_states.append(state.copy())
            # record_positions.append(agent_position.copy())

            # ==== Heading Calculation ====
            vec_to_target = target_position - agent_position
            dist_to_target = np.linalg.norm(vec_to_target)
            if dist_to_target < 1e-6:
                break

            # Base angle towards target
            angle_to_use = np.arctan2(vec_to_target[1], vec_to_target[0])
            r = min(2.0, dist_to_target)

            # ==== Extract nearest obstacle info (state_dim=10) ====
            barricade_state = state[6:8]
            direction = state[8]

            # If obstacle detected, adjust subgoal along tangent using direction
            if abs(direction) > 0:
                dx, dy = barricade_state

                if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
                    # fallback: small step along current heading
                    lc_target_x = agent_position[0] + r * np.cos(angle_to_use)
                    lc_target_y = agent_position[1] + r * np.sin(angle_to_use)
                else:
                    # Tangent vector: rotate (dx, dy) by 90° * direction
                    tangent_vec = np.array([dy, -dx]) * np.sign(direction)
                    tangent_vec /= np.linalg.norm(tangent_vec) + 1e-8       # normalize
                    lc_target_x = agent_position[0] + r * tangent_vec[0]
                    lc_target_y = agent_position[1] + r * tangent_vec[1]
            else:
                # Normal heading towards target
                lc_target_x = agent_position[0] + r * np.cos(angle_to_use)
                lc_target_y = agent_position[1] + r * np.sin(angle_to_use)

            # ==== Set low-level subgoal ====
            env.set_lc_target([lc_target_x, lc_target_y])

            # ==== Low-level Control ====
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]

            if lc_state_dim == 3:
                low_state = low_state[3:6]
            elif lc_state_dim == 6:
                low_state = low_state[:6]
            elif lc_state_dim == 11:
                low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

            low_action = ppo_agent_lc.select_action(low_state)
            low_action = env.clip_action(low_action)

            state, reward, terminated, truncated, info = env.step(low_action)

            done = terminated or truncated
            agent_position = env.agent_pos
            state_14 = env.state_14

            # ==== 保存 state ====
            if _save_state:
                episode_state.append({
                    "t": t,
                    "state": state.copy(),
                    "low_state": low_state.copy(),
                    "high_action": angle_to_use,
                    "low_action": low_action.tolist()
                })
            if _save_traj:
                episode_traj.append({
                    "step": t,
                    "agent_x": agent_position[0], "agent_y": agent_position[1],
                    "lc_target_x": lc_target_x, "lc_target_y": lc_target_y,
                    "high_action": angle_to_use, 
                    "low_action": low_action.tolist() if hasattr(low_action, 'tolist') else low_action,
                    "reward": reward, "done": done
                })

            t += 1
        # ==== Episode 结束 ====
        end_state = env.end_state
        end_angle = env.angle
        start_x, start_y = start_position
        target_x, target_y = env.target_position
        results.append([n, start_x, start_y, target_x, target_y, flow_id, end_state, end_angle, t])

        # ==== 保存 pickle 轨迹 ====
        if _save_traj:
            with open(traj_pickle_path, "ab") as f:
                pickle.dump(episode_traj, f)
                f.flush()
                os.fsync(f.fileno())

        # ==== 保存 HDF5 state 文件 ====
        if _save_state and len(episode_state) > 0:
            grp_name = f"episode_{n:05d}"
            if grp_name in state_h5_file:
                del state_h5_file[grp_name]
            grp = state_h5_file.create_group(grp_name)

            grp.create_dataset("t", data=np.array([s["t"] for s in episode_state]), compression="gzip")
            grp.create_dataset("state", data=np.array([s["state"] for s in episode_state]), compression="gzip")
            grp.create_dataset("low_state", data=np.array([s["low_state"] for s in episode_state]), compression="gzip")

            # high_action 可能为 None
            for s in episode_state:
                if s["high_action"] is not None:
                    high_action_dim = len(s["high_action"])
                    break
            else:
                high_action_dim = 2  # 默认值
            high_action_data = np.array([
                s["high_action"] if s["high_action"] is not None else [np.nan]*high_action_dim
                for s in episode_state
            ])
            grp.create_dataset("high_action", data=high_action_data, compression="gzip")

            # low_action
            low_action_data = np.array([
                np.atleast_1d(s["low_action"]) for s in episode_state
            ])
            grp.create_dataset("low_action", data=low_action_data, compression="gzip")

        # ==== 分类统计 ====
        if end_state == "success":
            success_times += 1
            count_time.append(env.step_counter * env.dt)
        elif end_state == "collide":
            collide_times += 1
        elif end_state == "outbound":
            outbound_times += 1
        elif end_state == "timelimit":
            timeout_times += 1

        # ==== 保存 CSV ====
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
        results.clear()

    env.close()
    if _save_state:
        state_h5_file.close()
        print(colored(f"Saved states to {state_h5_path}", 'cyan'))

    average_time = np.mean(count_time) if count_time else float('nan')
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print(f"State_dim: {state_dim_mode} || Average Navigation Time: {average_time}")
    print(colored(f"Saved test results to {csv_path}", 'cyan'))
    if _save_traj:
        print(colored(f"Saved trajectories to {traj_pickle_path}", 'cyan'))
###########################

def PathTracking_test_single(_id_in=1):
    ########################### Mode: Path Tracking (Single Trajectory) ###########################
    print("============================================================================================")

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #################################################

    ################ RL Hyperparameters ################
    has_continuous_action_space = True
    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0001
    lr_critic = 0.0002
    action_std_4_test = 0.04
    #################################################

    ################ Env Settings ################
    true_random = False
    random_seed = _id_in if not true_random else 0
    _is_head = False
    video_dir = "./Video_Tracking_Single"
    max_ll_steps = 5
    #################################################

    ########################### Task Settings ###########################
    test_mode = False
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

    max_steps = 250
    max_ep_len = max_steps + 20

    state_dim_mode = 10
    lc_state_dim = 6

    force_range = 15.0
    force_clip = 15.0

    _is_hrl = True
    if _is_hrl:
        ppo_path_lc = "./models/exp/lc_dim6.pth"
    else:
        ppo_path_lc = "./models/exp/rl/dim6/9083.pth"
    #################################################

    ########################### Environment ###########################
    flow_seed = 0
    env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=None, target_position=None,
        _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test=test_mode,
        _state_dim=state_dim_mode, _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip, _test_info=False,
        video_dir=video_dir, _near_mode=_is_near, _fov=_fov)

    ########################### Low-level PPO ###########################
    lc_action_dim = env.lc_action_space.shape[0]
    lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
    lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0

    ppo_agent_lc = Classic_PPO(
        lc_state_dim,
        lc_action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=action_std_4_test,
        continuous_action_output_scale=lc_action_output_scale,
        continuous_action_output_bias=lc_action_output_bias
    )
    ppo_agent_lc.load_full(ppo_path_lc)
    ppo_agent_lc.set_eval_mode(True)
    #################################################

    ########################### Single Trajectory ###########################
    n_points = 50
    R = 24
    center_x, center_y = 180, 64

    t = np.linspace(0, np.pi, n_points)
    x = center_x + R * np.cos(t)
    y = center_y + R * np.sin(t)
    traj_points = np.vstack((x, y)).T

    print(colored("Single trajectory generated.", "cyan"))
    #################################################

    ########################### Tracking Execution ###########################
    flow_id = 1

    state, info = env.reset(
        _flow_init=flow_id,
        _start_position_init=traj_points[0],
        _target_position_init=traj_points[1],
        _verbose=False
    )

    state_14 = env.state_14.copy()
    agent_position = env.agent_pos.copy()

    i = 1
    t_step = 0
    done = False

    near_threshold = env.max_detect_dis + env.window_r / 2

    print(colored("Start trajectory tracking...", "green"))

    while not done and t_step < max_ep_len:

        lc_target_x, lc_target_y = env.target_position
        env.set_lc_target([lc_target_x, lc_target_y])

        # ---------- Low-level state ----------
        low_state = state_14.copy()
        low_state[3] = lc_target_x - agent_position[0]
        low_state[4] = lc_target_y - agent_position[1]


        # if _is_hrl:
        #     low_state[3] = lc_target_x - agent_position[0]
        #     low_state[4] = lc_target_y - agent_position[1]
        # else:
        #     low_state[3] = (lc_target_x - agent_position[0]) / near_threshold
        #     low_state[4] = (lc_target_y - agent_position[1]) / near_threshold

        if lc_state_dim == 3:
            low_state = low_state[3:6]
        elif lc_state_dim == 6:
            low_state = low_state[:6]
        elif lc_state_dim == 11:
            low_state = np.hstack([low_state[3:6], state_14[6:14]])

        # ---------- Action ----------
        low_action = ppo_agent_lc.select_action(low_state)
        low_action = env.clip_action(low_action)

        # ---------- Step ----------
        state, reward, terminated, truncated, info = env.step(low_action)
        agent_position = env.agent_pos.copy()
        state_14 = env.state_14.copy()

        t_step += 1

        # ---------- Waypoint switch ----------
        if terminated:
            i += 1
            if i < len(traj_points):
                env.set_target(traj_points[i])
            else:
                done = True
                print(colored(f"Trajectory completed at step {t_step}", "green"))

        if truncated:
            print(colored(f"Truncated at step {t_step}", "red"))
            break

    print(colored("Tracking finished.", "blue"))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help="Enter 0 to run test(), 1 to run success_rate_test(), 2 to run success_rate_test_1()")
    parser.add_argument("id", type=int, nargs="?", default=0, help="Random seed for the test function (0 for fully random, otherwise fixed)")
    args = parser.parse_args()

    if args.mode == 0:
        test(args.id)
    elif args.mode == 1:
        success_rate_test_HRL(args.id)
    elif args.mode == 2:
        success_rate_test_RL(args.id)
    elif args.mode == 3:
        test_1(args.id)
    elif args.mode == 4:
        Hovering_test_HRL(args.id)
    elif args.mode == 5:
        Hovering_test_RL(args.id)
    elif args.mode == 6:
        Hovering_test_HRL_2(args.id)
    elif args.mode == 7:
        PathTracking_test_HRL(args.id)
    elif args.mode == 8:
        test_fixed(args.id)
    elif args.mode == 9:
        test_heading(args.id)
    elif args.mode == 10:
        success_rate_test_Heading(args.id)
    elif args.mode == 11:
        PathTracking_test_single(args.id)
