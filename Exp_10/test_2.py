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

import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO
import hjbppo
from hjbppo import HJB_PPO
import icm_ppo
from icm_ppo import ICM_PPO

from env_new import train_env_basic_2_2
from env_new import train_env_upper_2
from env_new import train_env_upper_2_1
from env_new import train_env_upper_2_3


# easy
ppo_path = './models/exp/rl_dim11_9070.pth'
ppo_path = './models/exp/rl_dim6.pth'
ppo_path_1 = './models/easy_task/0515_5.pth'


save_dir = './SuccessTest'
save_dir_1 = './model_output'
# plot_save_dir = './model_output/success_plot_withp.png'



def plot_success(all_position_records, success_flags, success_time, total_time, start_center, target_center, save_dir=None):
    """ Plot success/fail/collision trajectories """
    fig, ax = plt.subplots(figsize=(16, 9))
    circles = [
        {"center": (53, 32), "radius": 8},
        {"center": (53, 96), "radius": 8},
        {"center": (109, 64), "radius": 8}
    ]

    min_x = 0
    max_x = 20 * 16
    min_y = 0
    max_y = 8 * 16
    # edge margin
    x_margin = 2
    y_margin = 2
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    for circle in circles:
        ax.add_patch(
            plt.Circle(circle["center"], circle["radius"], color='red', alpha=0.3)
        )

    # random range
    start_circle = plt.Circle(start_center, 56, color='green', fill=False, linestyle='--')
    ax.add_artist(start_circle)
    target_circle = plt.Circle(target_center, 56, color='red', fill=False, linestyle='--')
    ax.add_artist(target_circle)

    # Line settings
    line_alpha = 0.8  # transparency
    line_width = 0.2

    # success_flag 1:success 2:collision 3:fail
    color_map = {1: 'green', 2: 'blue', 0: 'red'}

    for record, success_flag in zip(all_position_records, success_flags):
        x_values = [pos[0] for pos in record]
        y_values = [pos[1] for pos in record]
        line_color = color_map.get(success_flag, 'black')
        ax.plot(x_values, y_values, color=line_color, linestyle='-', linewidth=line_width, alpha=line_alpha)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_title(f'Success Rate:{success_time}/{total_time}')
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir, dpi=300)
    # plt.show()


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

def _save_trajectories(file_name, _trajectories):
    """
    save positions as ".csv"
    :param file_name:save path
    :param _trajectories: trajectory list(start_x, start_y, target_x, target_y)
    """
    if not _trajectories:
        print("No trajectories to save.")
        return
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_x", "start_y", "target_x", "target_y", "dis_2_target"])
        writer.writerows(_trajectories)
    print(f"trajectories saved to {file_name}")


def success_rate_test_RL(_seed_in=1):
    ########################### Mode 2: RL_test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 300
    max_ep_len = max_steps + 20
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    # state_dim_mode = 11
    state_dim_mode = 6
    # state_dim_mode = 3
    ppo_path = f'./models/exp/rl/dim{state_dim_mode}/9083.pth' # 9080 - 9084

    # set flow seed if required (0 : random flow; else : fixed flow)
    flow_seed = 0
    # flow_seed = 1

    # set random mode(True : real random mode; else : random.seed(random_seed))
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = _seed_in
    
    test_mode = False
    switch_range = 8
    
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
    env = train_env_basic_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
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
    # Load Position Flie
    test_time = 500

    # record parameters init
    success_times = 0
    collide_times = 0
    timeout_times = 0
    outbound_times = 0
    # navigation_time(success)
    count_time = []
    success_trajectories = []
    collide_trajectories = []
    outbound_trajectories = []
    timeout_trajectories = []
    # Test
    for n in range(test_time):
        # 1 env reset
        state, info = env.reset()
        print(f"Episode{n} Start")
        print("position_reset:", env.agent_pos)
        for i in range(1, max_ep_len + 1):
            # 2 env step
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, info= env.step(action)
            done = terminated or truncated
            end_state = env.end_state
            if done:      
                if end_state == "success":
                    print(colored(f"Episode{n} success", 'green'))
                    success_times += 1
                    # success_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                    count_time.append(env.step_counter * env.dt)
                    break
                elif end_state == "collide":
                    print(colored(f"Episode{n} collide", 'yellow'))
                    collide_times += 1
                    # collide_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                    break
                elif end_state == "outbound":
                    print(colored(f"Episode{n} outbound", 'red'))
                    outbound_times += 1
                    # outbound_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                    break
                elif end_state == "timelimit":
                    print(colored(f"Episode{n} timeout", 'blue'))
                    timeout_times += 1
                    # timeout_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                    break
        # ppo_agent.buffer.clear()
    env.close()

    average_time = np.mean(count_time)
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print(f"State_dim: {state_dim_mode} || Average Navigation Time: {average_time}")
    # name of ppo method
    # ppo_name = os.path.splitext(os.path.basename(ppo_path))[0]
    # success_name = f"{ppo_name}_success_random_{_id}.csv"
    # collide_name = f"{ppo_name}_collide_random_{_id}.csv"
    # outbound_name = f"{ppo_name}_outbound_random_{_id}.csv"
    # timeout_name = f"{ppo_name}_timeout_random_{_id}.csv"
    # success_filename = os.path.join(save_dir, success_name)
    # _save_trajectories(file_name=success_filename, _trajectories=success_trajectories)
    # collide_filename = os.path.join(save_dir, collide_name)
    # _save_trajectories(file_name=collide_filename, _trajectories=collide_trajectories)
    # outbound_filename = os.path.join(save_dir, outbound_name)
    # _save_trajectories(file_name=outbound_filename, _trajectories=outbound_trajectories)
    # timeout_filename = os.path.join(save_dir, timeout_name)
    # _save_trajectories(file_name=timeout_filename, _trajectories=timeout_trajectories)


def success_rate_test_HRL(_id_in=1):
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
    flow_seed = 0 
    test_mode = False
    switch_range = 8
    theta_mode = False
    switch_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    random_range = 32
    max_steps = 400
    max_ep_len = max_steps + 20
    state_dim_mode = 10
    lc_state_dim = 6
    speed_range = 15.0
    speed_clip = 15.0
    ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
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
    # speed_range = 20.0
    # speed_clip = 20.0
    # ppo_path = f'./models/exp/hrl/dim10/a20+fov270+near.pth'
    # ppo_path_1 = f'./models/exp/lc_dim14_a20.pth'

    
    ########### 无障碍任务 ###########
    # flow_seed = 0 
    # test_mode = False
    # switch_range = 4
    # theta_mode = False
    # switch_mode = False
    # _is_normalize = True
    # _is_avoid = False
    # _is_near = False
    # _start = np.array([240.0, 64.0])
    # _target = np.array([160.0, 64.0])
    # random_range = 32

    ##################################################### ///// ##################################################### 

    
    ########### Environment ########### 
    env = train_env_upper_2.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target,
        _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=speed_range, v_range=speed_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip,
        _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov = _fov
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
    test_time = 2000
    success_times = 0
    collide_times = 0
    timeout_times = 0
    outbound_times = 0
    save_interval = 100
    count_time = []

    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    subgoal_threshold = 1

    # ==== CSV 文件初始化 ====
    # csv_path = f"./Experiment/a20_test_seed_{random_seed}.csv"
    csv_path = f"./Experiment/a15_test_seed_{random_seed}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results = []

    # 如果是首次运行，写入表头
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Start_X", "Start_Y", "Target_X", "Target_Y", "Flow_ID", "End_State"])

    # ==== 主测试循环 ====
    for n in range(test_time):
        # Env Reset
        state, info = env.reset()
        state_14 = env.state_14
        agent_position = env.agent_pos
        start_position = agent_position
        print("position_reset:", agent_position)
        flow_id = env.flow_init

        t = 0
        hl_time_step = 0
        done = False
        while not done and t < max_ep_len:
            # === High-level policy ===
            high_state = state
            use_fallback = env.d_2_target < switch_range
            if use_fallback:
                vec_to_target = env.target_position - env.agent_pos
                target_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            else:
                high_action = ppo_agent.select_action(high_state)
                high_action = np.clip(high_action, env.low, env.high)
                hl_time_step += 1

            # === Planner Policy ===
            agent_position = env.agent_pos
            if _is_head:
                angle_to_use = target_angle if use_fallback else high_action
                if use_fallback:
                    r = np.linalg.norm(env.target_position - agent_position)
                else:
                    r = 2
                lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=angle_to_use, r=r)
            else:
                if use_fallback:
                    r = np.linalg.norm(env.target_position - agent_position)
                    angle_to_use = target_angle
                else:
                    r = 2
                    if not theta_mode:
                        vec = high_action
                        vec = vec / (np.linalg.norm(vec) + 1e-6)
                        angle_to_use = np.arctan2(vec[1], vec[0])
                    else:
                        angle_to_use = high_action
                lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=angle_to_use, r=r)

            # === Blind Policy ===
            # global_target = env.target_position
            # dx, dy = global_target[0] - agent_position[0], global_target[1] - agent_position[1]
            # angle_2_target = np.arctan2(dy, dx)
            # lc_target_x = agent_position[0] + r * np.cos(angle_2_target)
            # lc_target_y = agent_position[1] + r * np.sin(angle_2_target)

            env.set_lc_target([lc_target_x, lc_target_y])

            # === Low-level loop ===
            ll_step = 0
            done_lc = False
            while ll_step < max_ll_steps:
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
                    break

        # === Episode 结束 ===
        end_state = env.end_state
        start_x, start_y = start_position
        target_x, target_y = env.target_position
        results.append([n, start_x, start_y, target_x, target_y, flow_id, end_state])

        # === 分类统计 ===
        if end_state == "success":
            print(colored(f"Episode{n} success", 'green'))
            success_times += 1
            count_time.append(env.step_counter * env.dt)
        elif end_state == "collide":
            print(colored(f"Episode{n} collide", 'yellow'))
            collide_times += 1
        elif end_state == "outbound":
            print(colored(f"Episode{n} outbound", 'red'))
            outbound_times += 1
        elif end_state == "timelimit":
            print(colored(f"Episode{n} timeout", 'blue'))
            timeout_times += 1

        # === 每隔 save_interval 保存一次 ===
        if (n + 1) % save_interval == 0 or n == test_time - 1:
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)
            print(colored(f"[Partial results saved up to Episode {n}]", 'cyan'))
            results.clear()  # ✅ 清空缓存，避免重复写

    env.close()

    # === 结果统计 ===
    average_time = np.mean(count_time) if count_time else float('nan')
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print(f"State_dim: {state_dim_mode} || Average Navigation Time: {average_time}")
    print(colored(f"Saved test results to {csv_path}", 'cyan'))


def test(_id_in=1):
    ########################### Mode 0: HRL Single Test ###########################
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
    flow_seed = 1 
    test_mode = False
    switch_range = 8
    theta_mode = False
    _is_normalize = True
    _is_avoid = True
    _is_near = False
    _fov = np.pi
    switch_mode = False

    # _start_position = np.array([240.0, 64.0])
    # _target_position = np.array([180.0, 76.0])

    _start_position = None
    _target_position = None

    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])

    # switch_mode = True
    # _start = np.array([240.0, 64.0])
    # _target = np.array([160.0, 64.0])

    random_range = 32
    max_steps = 400
    max_ep_len = max_steps + 20
    state_dim_mode = 10
    lc_state_dim = 6
    speed_range = 15.0
    speed_clip = 15.0
    ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
    ppo_path_1 = f'./models/exp/lc_dim6.pth'
    video_dir = "./Video_10_Obstacle/video_frames"
    env = train_env_upper_2.foil_env(
        args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position = _start_position, target_position = _target_position,
        _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, 
        _init_flow_num=flow_seed,  _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode, 
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range,
        u_range=speed_range, v_range=speed_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip,
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
            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=angle_to_use, r=r)
        else:
            if use_fallback:
                r = np.linalg.norm(env.target_position - agent_position)
                angle_to_use = target_angle
            else:
                r = 2
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


def Hovering_test_HRL(_id_in=1):
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
    speed_range = 15.0
    speed_clip = 15.0
    ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
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
    save_path = os.path.join(save_dir, "hover_results.pkl")

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
    max_steps = 1200
    max_ep_len = max_steps + 20
    # max_ep_len = 200
    state_dim_mode = 10
    lc_state_dim = 6
    speed_range = 15.0
    speed_clip = 15.0
    ppo_path = f'./models/exp/hrl/dim10/a15+fov180+small.pth'
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

    ########################### Tracking Path ###########################
    a = 50   # 横向振幅（宽度的一半）
    b = 20   # 纵向振幅（高度的一半）
    n_points = 500  # 轨迹采样点数

    # 起点目标
    start_x = 180
    start_y = 72

    # 找一个起点相位（这里我们直接设 t=0 对应起点）
    t = np.linspace(0, 2*np.pi, n_points)

    # 原始曲线（相对于 0,0）
    x = a * np.sin(t)
    y = b * np.sin(2*t)

    # 平移到 (180, 72)
    x += start_x
    y += start_y

    points = np.vstack((x, y)).T
    arc_points = points
    ###################################################### 

    save_dir = "./tracking_test_results"
    os.makedirs(save_dir, exist_ok=True)

    flow_list = range(1, 25)
    subgoal_threshold = 1

    for flow_id in flow_list:
        print(colored(f"\n===== Testing Flow Field {flow_id} =====", "green"))

        results = []

        state, info = env.reset(
            _flow_init=flow_id,
            _start_position_init=arc_points[0],
            _target_position_init=arc_points[1],
            _verbose=False
        )
        state_14 = env.state_14
        agent_position = env.agent_pos.copy()
        i = 1
        t = 0
        done = False

        while not done and t < max_ep_len:
            target_position = env.target_position
            lc_target_x, lc_target_y = target_position
            env.set_lc_target([lc_target_x, lc_target_y])

            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]
            if lc_state_dim == 3:
                low_state = low_state[3:6]
            elif lc_state_dim == 6:
                low_state = low_state[:6]
            elif lc_state_dim == 11:
                low_state = np.hstack([low_state[3:6], state_14[6:14]]).astype(np.float32)

            low_action = ppo_agent_lc.select_action(low_state)
            cliped_low_action = env.clip_action(low_action)

            state, reward, terminated, truncated, info = env.step(low_action)
            agent_position = env.agent_pos.copy()
            state_14 = env.state_14.copy()

            results.append({
                "flow_id": flow_id,
                "step": t,
                "position": agent_position.copy(),
                "speed": env.speed,
                "velocity": env.agent_velocity.copy(),
                "agent_angle": env.angle,
                "omega": env.vel_angle,
                "action": cliped_low_action.copy(),
                "target_position": [lc_target_x, lc_target_y]
            })

            t += 1
            if terminated:
                i += 1
                if i < len(arc_points):
                    env.set_target(arc_points[i])
                else:
                    done = True
                    print(colored(f"All targets reached at step {t}", "green"))
            if truncated:
                print(colored(f"Truncated at step {t}", "red"))
                break

        # 每个流场测试完保存一次
        save_path = os.path.join(save_dir, f"controller_tracking_flow_{flow_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(colored(f"Flow {flow_id} data saved: {save_path}", "green"))

        # 清空结果，释放内存
        results.clear()

    env.close()
    print(colored("\nAll flows tested and saved individually.", "blue"))

    """"
    ########################### Test ###########################
    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    env.set_traj(arc_points)

    subgoal_threshold = 1
    flow_init = 1
    i = 1
    state, info = env.reset(_flow_init=None, _start_position_init=arc_points[0], _target_position_init=arc_points[1], _verbose=False)

    # Store initial states
    state_14 = env.state_14
    agent_position = env.agent_pos
    print("position_reset:", agent_position)
    

    # Main episode loop
    t = 0
    hl_time_step = 0
    done = False

    while not done and t < max_ep_len:
        target_position = env.target_position
        lc_target_x, lc_target_y = target_position
        # === Normal Subgoal ===
        env.set_lc_target([lc_target_x, lc_target_y])
        print(f"=== Subgoal: ({lc_target_x}, {lc_target_y}) ===")
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
        print(f"=== Low Level Step {t}, Action: {low_action}")
        print(f"===               -> Clip_Action: {cliped_low_action} ===")
        # Step env
        state, reward, terminated, truncated, info = env.step(low_action)
        agent_position = env.agent_pos
        state_14 = env.state_14
        t += 1
        if terminated:
            i += 1
            env.set_target(arc_points[i])
        if truncated:
            break
    env.close()
    print("Total_Steps:", t)
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], help="Enter 0 to run test(), 1 to run success_rate_test(), 2 to run success_rate_test_1()")
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
