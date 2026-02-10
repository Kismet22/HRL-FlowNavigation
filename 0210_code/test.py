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
from env_new import train_env
from env_new import train_env_plus
from env_new import train_env_basic_2_2

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
def test(control_mode="HRL", _id_in=1, TASK_MODE="avoid"):
    print("============================================================================================")
    print(f"[TEST MODE] control_mode={control_mode}, TASK_MODE={TASK_MODE}, seed={_id_in}")

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
    true_random = True
    random_seed = 0 if true_random else _id_in
    _is_head = False
    max_ll_steps = 5
    #################################################

    ##################################################### Task Selection #####################################################
    flow_seed = _id_in
    test_mode = True
    theta_mode = False
    _is_normalize = True
    _is_near = False
    _fov = np.pi
    switch_mode = False

    # --- Task mode switch ---
    switch_range = 8
    if TASK_MODE == "avoid":
        _is_avoid = True
        random_range = 32
        max_steps = 600
        _start = np.array([220.0, 64.0])
        _target = np.array([64.0, 64.0])
        video_dir = "./0201_EXP_Obstacle/video_frames"
        ppo_path_1 = f'./models/exp/lc_dim6.pth'
    elif TASK_MODE == "free":
        _is_avoid = False
        random_range = 32
        max_steps = 600
        _start = np.array([240.0, 64.0])
        _target = np.array([160.0, 64.0])
        video_dir = "./EXP_Free/video_frames"
    else:
        raise ValueError(f"Unknown TASK_MODE: {TASK_MODE}")

    max_ep_len = max_steps + 20
    force_range = 15.0
    force_clip = 15.0

    ################ Model Paths ################
    state_dim_mode = 10
    _model_id = 1
    ppo_path = f'./models/exp/hrl/avoid/dim{state_dim_mode}/a15+fov180_{_model_id}.pth'
    r_test = 2

    # state_dim_mode = 6
    # # _model_id = 9210
    # # ppo_path = f'./models/exp/hrl/joint/dim{state_dim_mode}/{_model_id}.pth'
    # _model_id = 5
    # ppo_path = f'./models/exp/hrl/free/dim{state_dim_mode}_switch/{_model_id}.pth'
    # r_test = 5

    if control_mode == "HEADING":
        r_test = 2
        max_ll_steps = 1

    lc_state_dim = 6
    # if TASK_MODE == "avoid":
    #     ppo_path_1 = f'./models/exp/lc_dim6.pth'
    # else:
    #     ppo_path_1 = f"./models/exp/lc/dim{lc_state_dim}/{_model_id}.pth"
    ppo_path_1 = f'./models/exp/lc_dim6.pth'
    # video_dir = "./EXP_HRL/video_frames"

    ################ Environment ################
    env = train_env.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=None, target_position=None, _include_flow=True,
        _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode,
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range, u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip, _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov, _plot_task = "navigation")

    ################ PPO Agents ################
    ppo_agent = Classic_PPO(env.observation_space.shape[0], env.action_space.shape[0], lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test, continuous_action_output_scale=(env.action_space.high - env.action_space.low) / 2.0, continuous_action_output_bias=(env.action_space.high + env.action_space.low) / 2.0)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)

    ppo_agent_lc = Classic_PPO(lc_state_dim, env.lc_action_space.shape[0], lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test, continuous_action_output_scale=(env.lc_action_space.high - env.lc_action_space.low) / 2.0, continuous_action_output_bias=(env.lc_action_space.high + env.lc_action_space.low) / 2.0)
    ppo_agent_lc.load_full(ppo_path_1)
    ppo_agent_lc.set_eval_mode(True)

    ################ Test Initialization ################
    # x_start = 173
    # y_start = 81
    # x_target = 19.3
    # y_target = 33.3
    x_start = 232.68
    y_start = 56.88
    x_target = 74.37
    y_target = 53.06
    # _extra_obs = False
    _extra_obs = True

    _start_position_init = np.array([x_start, y_start])
    _target_position_init = np.array([x_target, y_target])
    # _start_position_init = None
    # _target_position_init = None
    state, info = env.reset(_start_position_init=_start_position_init, _target_position_init=_target_position_init, _add_virtual_cylinders=_extra_obs, _custom_cylinders=[(160, 64), (160, 108), (160, 16)])
    # state, info = env.reset()
    state_14 = env.state_14
    agent_position = env.agent_pos
    target_position = env.target_position

    print("position_reset:", agent_position)

    record_high_states = []
    record_positions = []

    t = 0
    hl_time_step = 0
    done = False

    ################ Main Episode Loop ################
    while not done and t < max_ep_len:
        high_state = state

        # Record states for analysis
        record_high_states.append(high_state.copy())
        record_positions.append(agent_position.copy())

        # ================= High-Level Decision =================
        use_fallback = env.d_2_target < switch_range

        if control_mode == "HEADING":
            vec_to_target = target_position - agent_position
            angle_to_use = np.arctan2(vec_to_target[1], vec_to_target[0])
            r = min(r_test, np.linalg.norm(vec_to_target))
            print(colored("=== HEADING mode: reactive replanning ===", "blue"))

        else:
            if use_fallback:
                vec_to_target = target_position - agent_position
                angle_to_use = np.arctan2(vec_to_target[1], vec_to_target[0])
                r = np.linalg.norm(vec_to_target)
                print(colored(
                    f"=== Fallback to target mode | d2target = {env.d_2_target:.2f} ===",
                    "yellow"
                ))
            else:
                high_action = ppo_agent.select_action(high_state)
                high_action = np.clip(high_action, env.low, env.high)
                hl_time_step += 1
                print(colored(
                    f"=== High Level Step {hl_time_step}, Action: {high_action} ===",
                    "blue"
                ))

                if not theta_mode:
                    vec = high_action / (np.linalg.norm(high_action) + 1e-6)
                    angle_to_use = np.arctan2(vec[1], vec[0])
                else:
                    angle_to_use = high_action

                r = r_test

        # ================= Subgoal Computation =================
        lc_target_x, lc_target_y = compute_subgoal(
            pos=agent_position,
            angle=angle_to_use,
            r=r
        )
        env.set_lc_target([lc_target_x, lc_target_y])

        print(
            f"=== Subgoal: ({lc_target_x:.2f}, {lc_target_y:.2f}) | "
            f"Angle used: {angle_to_use:.2f} ==="
        )

        # ================= Low-Level Control Loop =================
        ll_step = 0
        done_lc = False

        while ll_step < max_ll_steps:
            agent_angle_old = env.angle

            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]
            low_state = low_state[:6]

            low_action = ppo_agent_lc.select_action(low_state)
            cliped_low_action = env.clip_action(low_action)

            print(f"=== Low Level Step {t}, Action: {low_action}")
            print(f"===               -> Clip_Action: {cliped_low_action} ===")

            state, reward, terminated, truncated, info = env.step(low_action)
            done = terminated or truncated

            agent_position = env.agent_pos
            state_14 = env.state_14

            dist_to_subgoal = np.linalg.norm(
                agent_position - np.array([lc_target_x, lc_target_y])
            )

            if dist_to_subgoal < 1.0 or done or ll_step + 1 >= max_ll_steps:
                done_lc = True

            t += 1
            ll_step += 1

            if done_lc:
                ppo_agent_lc.buffer.clear()
                break

        ppo_agent.buffer.clear()

    ################ Cleanup ################
    env.close()
    print("Total_Steps:", t)

    ################ Save Records ################
    save_dir = "./Experiment/Single_test/"
    os.makedirs(save_dir, exist_ok=True)

    record_high_states = np.array(record_high_states)
    record_positions = np.array(record_positions)

    np.save(
        os.path.join(save_dir, f"record_high_states_dim_{state_dim_mode}.npy"),
        record_high_states
    )
    np.save(
        os.path.join(save_dir, f"record_positions_dim_{state_dim_mode}.npy"),
        record_positions
    )

    print(f"Saved: {save_dir}record_high_states_dim_{state_dim_mode}.npy")
    print(f"Saved: {save_dir}record_positions_dim_{state_dim_mode}.npy")
###########################

########################### Mode 2: Multi-Test ###########################
def success_rate_test(control_mode="HRL", _id_in=1, TASK_MODE="avoid"):
    """
    control_mode: "HRL" | "HEADING" | "JOINT"
    TASK_MODE: "avoid" | "free"
    """
    assert control_mode in ["HRL", "HEADING", "JOINT"]
    assert TASK_MODE in ["avoid", "free"]

    print("============================================================================================")
    print(f"==== Control Mode: {control_mode} | Task Mode: {TASK_MODE} ====")

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

    ################ Env Settings ################
    true_random = False
    random_seed = 0 if true_random else _id_in
    np.random.seed(random_seed)

    _is_head = False
    video_dir = "./Video_HRL/video_frames"
    max_ll_steps = 5
    if control_mode == "Heading":
        max_ll_steps = 1

    
    _save_traj = True
    # _save_traj = False
    _save_state = False
    # _extra_obs = False
    _extra_obs = True

    # test_mode = True if _save_traj else False
    test_mode = False
    # flow_seed = 15 if _save_traj else 0
    flow_seed = 1 if _save_traj else 0

    switch_range = 4 if _save_traj else 8 
    theta_mode = False
    switch_mode = False
    _is_normalize = True
    _is_near = False
    _center_mode = False
    # _fov = np.pi
    # _fov_angle = 180 # best id 1
    # model_id = 1
    # _fov = np.pi/2
    # _fov_angle = 90 # best id 3
    # model_id = 3
    _fov = 3*np.pi/2
    _fov_angle = 270 # best id 3
    model_id = 3

    ################ Task-specific Settings ################
    if TASK_MODE == "avoid":
        _is_avoid = True
        _start = np.array([220.0, 64.0])
        _target = np.array([64.0, 64.0])
        random_range = 32
        # max_steps = 600
        max_steps = 750
        state_dim_mode = 10
        # model_id = 1
        r_test = 2
        if _center_mode:
            ppo_path_hl = f"./models/exp/hrl/avoid/dim{state_dim_mode}/a15+fov{_fov_angle}_self_{model_id}.pth"
        else:
            ppo_path_hl = f"./models/exp/hrl/avoid/dim{state_dim_mode}/a15+fov{_fov_angle}_{model_id}.pth"
        # r_test = 3.5
        # ppo_path_hl = f"./models/exp/hrl/avoid/dim{state_dim_mode}_r_3/{model_id}.pth"
        # r_test = 5
        # ppo_path_hl = f"./models/exp/hrl/avoid/dim{state_dim_mode}_r_5/{model_id}.pth"
        
    elif TASK_MODE == "free":
        _is_avoid = False
        # ====== upstream ====== 
        _start = np.array([240.0, 64.0])
        _target = np.array([160.0, 64.0])
        max_steps = 300
        random_range = 32

        # ====== downstream ======
        # _start = np.array([160.0, 64.0])
        # _target = np.array([240.0, 64.0])
        # max_steps = 200
        # random_range = 32

        # ====== cross stream ======
        # _start = np.array([220.0, 32.0])
        # _target = np.array([220.0, 96.0])
        # _start = np.array([220.0, 96.0])
        # _target = np.array([220.0, 32.0])
        # max_steps = 200
        # random_range = 24

        if _save_traj:
            max_steps = 400

        state_dim_mode = 6
        if control_mode == "JOINT":
            model_id = 9213 # 9210 - 9214
            ppo_path_hl = f"./models/exp/hrl/joint/dim{state_dim_mode}/{model_id}.pth"
        else:
            model_id = 5
            ppo_path_hl = f"./models/exp/hrl/free/dim{state_dim_mode}_switch/{model_id}.pth"
        r_test = 5
        # ppo_path_hl = f"./models/exp/hrl/free/dim{state_dim_mode}/{model_id}.pth"
        # r_test = 2

    max_ep_len = max_steps + 20
    _start_position = None
    _target_position = None

    lc_state_dim = 6
    force_range = 15.0
    force_clip = 15.0

    # Model Path
    if control_mode == "JOINT":
        ppo_path_lc = f"./models/exp/lc/dim{lc_state_dim}/{model_id}.pth"
    else:
        ppo_path_lc = f"./models/exp/lc_dim{lc_state_dim}.pth"

    ################ Environment ####################
    env = train_env_plus.foil_env(
        args_1, max_step=max_steps,
        start_center=_start, target_center=_target,
        start_position=_start_position, target_position=_target_position,
        _include_flow=False, _plot_flow=False, _proccess_flow=False,
        _random_range=random_range, _init_flow_num=flow_seed,
        _pos_normalize=_is_normalize, _is_test=test_mode,
        _state_dim=state_dim_mode, _is_random=true_random,
        _set_random=random_seed, _switch_range=switch_range,
        u_range=force_range, v_range=force_range,
        _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode,
        u_clip=force_clip, v_clip=force_clip,
        _test_info=False, video_dir=video_dir,
        _near_mode=_is_near, _fov=_fov, _self_center=_center_mode
    )

    ################ Action Spaces ################
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lc_action_dim = env.lc_action_space.shape[0]

    action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
    action_output_bias = (env.action_space.high + env.action_space.low) / 2.0

    lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
    lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0

    ################ PPO ################
    ppo_agent_hl = Classic_PPO(
        state_dim, action_dim, lr_actor, lr_critic,
        gamma, K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test,
        continuous_action_output_scale=action_output_scale,
        continuous_action_output_bias=action_output_bias
    )
    ppo_agent_hl.load_full(ppo_path_hl)
    ppo_agent_hl.set_eval_mode(True)

    ppo_agent_lc = Classic_PPO(
        lc_state_dim, lc_action_dim, lr_actor, lr_critic,
        gamma, K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test,
        continuous_action_output_scale=lc_action_output_scale,
        continuous_action_output_bias=lc_action_output_bias
    )
    ppo_agent_lc.load_full(ppo_path_lc)
    ppo_agent_lc.set_eval_mode(True)

    ################ Save Path ################
    os.makedirs("./Experiment", exist_ok=True)
    if _center_mode:
        csv_path = f"./Experiment/{control_mode}_{TASK_MODE}_a15_test_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}_r_{r_test}_fov_{_fov_angle}_center.csv"
    else:
        csv_path = f"./Experiment/{control_mode}_{TASK_MODE}_a15_test_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}_r_{r_test}_fov_{_fov_angle}.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode","Start_X","Start_Y","Target_X","Target_Y","Flow_ID","End_State","End_angle","End_time"])

    if _save_traj:
        traj_pickle_path = f"./Experiment/{control_mode}_{TASK_MODE}_trajectories_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}_r_{r_test}_fov_{_fov_angle}.pkl"

    if _save_state:
        state_h5_path = f"./Experiment/{control_mode}_{TASK_MODE}_states_seed_{random_seed}_dim_{state_dim_mode}_model_{model_id}.h5"
        state_h5_file = h5py.File(state_h5_path, "a")

    ################ Sub-goal ################
    def get_subgoal_HRL(state, agent_position):
        use_fallback = env.d_2_target < 8
        if use_fallback:
            high_action = ppo_agent_hl.select_action(state)
            vec = env.target_position - agent_position
            angle = np.arctan2(vec[1], vec[0])
        else:
            high_action = ppo_agent_hl.select_action(state)
            vec = high_action / (np.linalg.norm(high_action) + 1e-6)
            angle = np.arctan2(vec[1], vec[0])
        r = np.linalg.norm(env.target_position - agent_position) if use_fallback else r_test
        x = agent_position[0] + r * np.cos(angle)
        y = agent_position[1] + r * np.sin(angle)
        return x, y, high_action

    def get_subgoal_HEADING(state, agent_position):
        """
        Compute subgoal for heading-based navigation.
        
        - If state_dim >= 9:
            Use barricade direction and turning sign for obstacle-aware subgoal.
        - If state_dim < 9:
            Pure heading subgoal without obstacle avoidance.
        """

        # ---------- Basic heading toward target ----------
        vec = env.target_position - agent_position
        angle = np.arctan2(vec[1], vec[0])
        r = min(2.0, np.linalg.norm(vec))

        # ---------- Check whether obstacle info exists ----------
        use_obstacle_info = len(state) >= 9

        if use_obstacle_info:
            barricade = state[6:8]
            direction = state[8]

            if abs(direction) > 1e-6 and np.linalg.norm(barricade) > 1e-6:
                dx, dy = barricade

                # Tangent direction determined by turning sign
                tangent = np.array([dy, -dx]) * np.sign(direction)
                tangent /= np.linalg.norm(tangent) + 1e-8

                x = agent_position[0] + r * tangent[0]
                y = agent_position[1] + r * tangent[1]

                return x, y, angle

        # ---------- Fallback: pure heading ----------
        x = agent_position[0] + r * np.cos(angle)
        y = agent_position[1] + r * np.sin(angle)

        return x, y, angle


    ################ Test Loop ################
    test_time = 20 if _save_traj else 2000
    subgoal_threshold = 1

    success_times = collide_times = timeout_times = outbound_times = 0
    count_time = []

    for n in range(test_time):
        state, info = env.reset(
            _add_virtual_cylinders=_extra_obs,
            _custom_cylinders=[(160, 64),(160,108),(160,16)]
        )

        state_14 = env.state_14
        agent_position = env.agent_pos
        start_position = np.copy(agent_position)
        t = 0
        done = False
        episode_traj = []
        episode_state = []
        results = []

        while not done and t < max_ep_len:
            if control_mode == "HRL":
                lc_x, lc_y, high_action = get_subgoal_HRL(state, agent_position)
            else:
                lc_x, lc_y, high_action = get_subgoal_HEADING(state, agent_position)

            env.set_lc_target([lc_x, lc_y])

            ll_step = 0
            while ll_step < max_ll_steps:
                low_state = state_14.copy()
                low_state[3] = lc_x - agent_position[0]
                low_state[4] = lc_y - agent_position[1]
                low_state = low_state[:6]

                low_action = ppo_agent_lc.select_action(low_state)
                low_action = env.clip_action(low_action)

                state, reward, terminated, truncated, info = env.step(low_action)
                done = terminated or truncated
                agent_position = env.agent_pos
                state_14 = env.state_14

                if _save_state:
                    episode_state.append({
                        "t": t,
                        "state": state.copy(),
                        "low_state": low_state.copy(),
                        "high_action": high_action,
                        "low_action": low_action.tolist()
                    })

                if _save_traj:
                    episode_traj.append({
                        "step": t,
                        "agent_x": agent_position[0], "agent_y": agent_position[1],
                        "lc_target_x": lc_x, "lc_target_y": lc_y,
                        "high_action": high_action,
                        "low_action": low_action.tolist() if hasattr(low_action,'tolist') else low_action,
                        "reward": reward, "done": done
                    })

                dist = np.linalg.norm(agent_position - np.array([lc_x, lc_y]))
                t += 1
                ll_step += 1
                if dist < subgoal_threshold or done:
                    break

        end_state = env.end_state
        end_angle = env.angle
        start_x, start_y = start_position
        target_x, target_y = env.target_position
        results.append([n,start_x,start_y,target_x,target_y,env.flow_init,end_state,end_angle,t])

        with open(csv_path,"a",newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
        results.clear()

        if _save_traj:
            with open(traj_pickle_path,"ab") as f:
                pickle.dump(episode_traj,f)
                f.flush()
                os.fsync(f.fileno())

        if _save_state and len(episode_state) > 0:
            grp_name = f"episode_{n:05d}"
            if grp_name in state_h5_file:
                del state_h5_file[grp_name]
            grp = state_h5_file.create_group(grp_name)
            grp.create_dataset("t", data=np.array([s["t"] for s in episode_state]), compression="gzip")
            grp.create_dataset("state", data=np.array([s["state"] for s in episode_state]), compression="gzip")
            grp.create_dataset("low_state", data=np.array([s["low_state"] for s in episode_state]), compression="gzip")
            high_action_dim = 2 if all(s["high_action"] is None for s in episode_state) else len([s for s in episode_state if s["high_action"] is not None][0])
            high_action_data = np.array([s["high_action"] if s["high_action"] is not None else [np.nan]*high_action_dim for s in episode_state])
            grp.create_dataset("high_action", data=high_action_data, compression="gzip")
            low_action_data = np.array([np.atleast_1d(s["low_action"]) for s in episode_state])
            grp.create_dataset("low_action", data=low_action_data, compression="gzip")

        if end_state=="success":
            success_times += 1
            count_time.append(env.step_counter*env.dt)
        elif end_state=="collide":
            collide_times += 1
        elif end_state=="outbound":
            outbound_times += 1
        elif end_state=="timelimit":
            timeout_times += 1

    env.close()
    if _save_state:
        state_h5_file.close()
        print(f"Saved states to {state_h5_path}")

    print(f"==== {control_mode} Results ====")
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print(f"Average Time: {np.mean(count_time) if count_time else float('nan')}")
    print(f"Saved test results to {csv_path}")
    if _save_traj:
        print(f"Saved trajectories to {traj_pickle_path}")
###########################

########################### Mode 3: RL_test ###########################
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
    _save_traj = True
    # _save_traj = False
    _save_state = False
    _extra_obs = False
    if _save_traj:
        flow_seed = 15
        # test_mode = True
        test_mode = False
        switch_range = 4
    else:
        flow_seed = 0
        test_mode = False
        switch_range = 8
    switch_mode = False
    _is_normalize = True
    _fov = np.pi
    _start_position = None
    _target_position = None
    # _start = np.array([220.0, 32.0])
    # _target = np.array([220.0, 96.0])
    # _start = np.array([220.0, 96.0])
    # _target = np.array([220.0, 32.0])
    # random_range = 24
    
    # _start = np.array([240.0, 64.0])
    # _target = np.array([160.0, 64.0])
    _start = np.array([160.0, 64.0])
    _target = np.array([240.0, 64.0])
    random_range = 32
    max_steps = 200
    max_ep_len = max_steps + 20
    input_max = 15.0
    clip_max = 15.0
    # _id = 9086
    # state_dim_mode = 11 # 9070 - 9074
    state_dim_mode = 6 # 9080 - 9084 best:9083
    # state_dim_mode = 3 # 9080 - 9082 9085 9086
    # ppo_path = f'./models/exp/rl/dim{state_dim_mode}/{_id}.pth'
    _id = 9213
    ppo_path = f'./models/exp/rl/dim{state_dim_mode}_switch/{_id}.pth'
 
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

########################### Mode 1: RL Single Test ###########################
def test_1(_id_in=1):
    print("============================================================================================")
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
###########################

########################### Mode 4: Multi-task ###########################
def test_m(control_mode="HRL", _id_in=1, TASK_MODE="avoid"):
    print("============================================================================================")
    print(f"[TEST MODE] control_mode={control_mode}, TASK_MODE={TASK_MODE}, seed={_id_in}")

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
    true_random = True
    random_seed = 0 if true_random else _id_in
    _is_head = False
    max_ll_steps = 5
    #################################################

    ##################################################### Task Selection #####################################################
    flow_seed = _id_in
    test_mode = True
    theta_mode = False
    _is_normalize = True
    _is_near = False
    _fov = np.pi
    switch_mode = False

    # --- Task mode switch ---
    switch_range = 8
    if TASK_MODE == "avoid":
        _is_avoid = True
        random_range = 32
        max_steps = 1600
        _start = np.array([220.0, 64.0])
        _target = np.array([64.0, 64.0])
    elif TASK_MODE == "free":
        _is_avoid = False
        random_range = 32
        max_steps = 200
        _start = np.array([240.0, 64.0])
        _target = np.array([160.0, 64.0])
    else:
        raise ValueError(f"Unknown TASK_MODE: {TASK_MODE}")

    max_ep_len = max_steps + 20
    force_range = 15.0
    force_clip = 15.0

    ################ Model Paths ################
    state_dim_mode = 10
    _model_id = 1
    ppo_path = f'./models/exp/hrl/avoid/dim{state_dim_mode}/a15+fov180_{_model_id}.pth'
    r_test = 2

    # state_dim_mode = 6
    # _model_id = 5
    # ppo_path = f'./models/exp/hrl/free/dim{state_dim_mode}_switch/{_model_id}.pth'
    # r_test = 5

    if control_mode == "HEADING":
        r_test = 2
        max_ll_steps = 1

    lc_state_dim = 6
    ppo_path_1 = f'./models/exp/lc_dim6.pth'
    video_dir = "./EXP_Setup/video_frames"

    ################ Environment ################
    env = train_env_plus.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=None, target_position=None, _include_flow=True,
        _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode,
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range, u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip, _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov, plot_mode="simple")

    ################ PPO Agents ################
    ppo_agent = Classic_PPO(env.observation_space.shape[0], env.action_space.shape[0], lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test, continuous_action_output_scale=(env.action_space.high - env.action_space.low) / 2.0, continuous_action_output_bias=(env.action_space.high + env.action_space.low) / 2.0)
    ppo_agent.load_full(ppo_path)
    ppo_agent.set_eval_mode(True)

    ppo_agent_lc = Classic_PPO(lc_state_dim, env.lc_action_space.shape[0], lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test, continuous_action_output_scale=(env.lc_action_space.high - env.lc_action_space.low) / 2.0, continuous_action_output_bias=(env.lc_action_space.high + env.lc_action_space.low) / 2.0)
    ppo_agent_lc.load_full(ppo_path_1)
    ppo_agent_lc.set_eval_mode(True)

    ################ Trajectory Function ################
    def generate_double_arc_traj_fixed_radius(P_start, P_goal, max_step_dist=2.0):
        """
        Generate trajectory: two half-circles (down then up) connecting P_start to P_goal.
        The radius is automatically D/4, and circle centers are calculated geometrically.
        """
        P_start = np.array(P_start, dtype=float)
        P_goal = np.array(P_goal, dtype=float)

        # Total distance and radius
        D = np.linalg.norm(P_goal - P_start)
        R = D / 4

        # Determine if horizontal or vertical alignment
        horizontal = abs(P_start[1] - P_goal[1]) < 1e-6

        traj = []

        if horizontal:
            # Horizontal case: y is constant
            # Circle centers
            center1 = P_start - np.array([R, 0])  # first circle down
            center2 = P_goal + np.array([R, 0])  # second circle up

            # First half-circle (down): from P_start to midpoint
            n1 = max(int(np.pi * R / max_step_dist), 2)
            for i in range(n1):
                theta = np.pi * i / (n1 - 1)  # 0 -> pi
                x = center1[0] + R * np.cos(theta)
                y = center1[1] - R * np.sin(theta)  # negative for down
                traj.append([x, y])

            # Second half-circle (up): from midpoint to P_goal
            n2 = max(int(np.pi * R / max_step_dist), 2)
            for i in range(n2):
                theta = np.pi * i / (n2 - 1)  # 0 -> pi
                x = center2[0] + R * np.cos(theta)
                y = center2[1] + R * np.sin(theta)  # positive for up
                traj.append([x, y])

        traj = np.array(traj)
        return traj

    ################ Reset ################

    # ===== Key positions =====
    _hover_position = np.array([233.0, 81.0])      # Phase 0 & 1 起点（悬停点）
    _phase2_start   = np.array([173.0, 81.0])      # Phase 2 起点
    _phase2_target  = np.array([19.3, 33.3])       # 最终目标

    hov_points = [_hover_position, _phase2_start]
    traj_points = generate_double_arc_traj_fixed_radius(_hover_position, _phase2_start, max_step_dist=2.0)

    state, info = env.reset(
        _start_position_init=_hover_position,
        _target_position_init=_phase2_target
    )

    state_14 = env.state_14
    agent_position = env.agent_pos
    target_position = env.target_position

    env.set_hover(hov_points)
    env.set_traj(traj_points)

    ################ Phase Settings ################
    task_phase = 0       # 0: hover, 1: trajectory, 2: planner
    phase_step = 0
    hover_steps = 50


    # ==== Test ===
    t = 0
    done = False
    _count = 0

    while not done and t < max_ep_len:

        # ================= Phase 0: Hover =================
        if task_phase == 0:
            agent_position = env.agent_pos
            lc_target_x, lc_target_y = _hover_position
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]
            
            env.set_lc_target([lc_target_x, lc_target_y])

            low_state = low_state[:6]

            low_action = ppo_agent_lc.select_action(low_state)
            state, reward, terminated, truncated, info = env.step(low_action)

            agent_position = env.agent_pos
            state_14 = env.state_14

            t += 1
            phase_step += 1

            if phase_step >= hover_steps:
                print(colored("=== Enter Trajectory Tracking Phase ===", "cyan"))
                task_phase = 1
                phase_step = 0
                _count = 0
            continue

        # ================= Phase 1: Trajectory =================
        if task_phase == 1:
            traj_reach_th = 1.0  # waypoint 到达阈值
            # ----- 当前 waypoint -----
            lc_target_x, lc_target_y = traj_points[phase_step]
            env.set_lc_target([lc_target_x, lc_target_y])

            # ----- low-level state -----
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]
            low_state = low_state[:6]

            # ----- low-level action -----
            low_action = ppo_agent_lc.select_action(low_state)
            state, reward, terminated, truncated, info = env.step(low_action)

            # ----- update -----
            agent_position = env.agent_pos
            state_14 = env.state_14
            t += 1

            # ----- waypoint reached? -----
            if np.linalg.norm(agent_position - traj_points[phase_step]) < traj_reach_th:
                phase_step += 1   # ✅ advance only when reached

            # ----- transition to Phase 2 -----
            if phase_step >= len(traj_points):
                _count += 1
                phase_step = len(traj_points) - 1
            if _count > 20:
                task_phase = 2
                phase_step = 0
                print(colored("=== Enter Planner-based Avoidance Phase ===", "green"))
            continue

        # ================= Phase 2: Planner / HRL =================
        high_state = state
        use_fallback = env.d_2_target < switch_range

        if use_fallback:
            vec = target_position - agent_position
            angle_to_use = np.arctan2(vec[1], vec[0])
            r = np.linalg.norm(vec)
        else:
            high_action = ppo_agent.select_action(high_state)
            high_action = np.clip(high_action, env.low, env.high)
            vec = high_action / (np.linalg.norm(high_action) + 1e-6)
            angle_to_use = np.arctan2(vec[1], vec[0])
            r = r_test

        lc_target_x, lc_target_y = compute_subgoal(agent_position, angle_to_use, r)
        env.set_lc_target([lc_target_x, lc_target_y])

        ll_step = 0
        while ll_step < max_ll_steps:
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]
            low_state = low_state[:6]

            low_action = ppo_agent_lc.select_action(low_state)
            state, reward, terminated, truncated, info = env.step(low_action)

            agent_position = env.agent_pos
            state_14 = env.state_14
            done = terminated or truncated

            t += 1
            ll_step += 1

            if done:
                break

        ppo_agent.buffer.clear()
        ppo_agent_lc.buffer.clear()


    env.close()
    print("Total steps:", t)

########################### Mode 5: hovering task ###########################
def test_hover_only(_id_in=1):
    print("============================================================================================")
    print(f"========== HOVER ONLY TEST, seed={_id_in} ==========")

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
    true_random = True
    random_seed = 0 if true_random else _id_in
    _is_head = False
    max_ll_steps = 5
    #################################################

    ##################################################### Task Selection #####################################################
    flow_seed = _id_in
    test_mode = True
    theta_mode = False
    _is_normalize = True
    _is_near = False
    _fov = np.pi
    switch_mode = False
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    random_range = 32
    state_dim_mode = 6
    switch_range = 8
    force_clip = force_range = 15.0
    _is_avoid = False

    ################ Basic settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, _ = parser_1.parse_known_args()
    args_1.action_interval = 10

    ################ Env settings ################
    max_steps = 250
    _hover_position = np.array([180.0, 32.0])
    # _hover_pos = [180.0, 32.0]
    video_dir = "./0201_EXP_Hovering/video_frames"

    env = train_env.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=None, target_position=None, _include_flow=True,
        _plot_flow=True, _proccess_flow=False, _plot_task = "hovering", _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode,
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range, u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip, _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov, plot_mode="full")

    ################ LC PPO ################
    lc_state_dim = 6
    ppo_agent_lc = Classic_PPO(
        lc_state_dim,
        env.lc_action_space.shape[0],
        lr_actor=1e-4,
        lr_critic=2e-4,
        gamma=0.99,
        K_epochs=40,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std_init=0.04,
        continuous_action_output_scale=(env.lc_action_space.high - env.lc_action_space.low) / 2.0,
        continuous_action_output_bias=(env.lc_action_space.high + env.lc_action_space.low) / 2.0
    )
    # ppo_agent_lc.load_full("./models/exp/lc_dim6.pth")
    ppo_agent_lc.load_full(f'./models/exp/rl/dim6/9083.pth')
    ppo_agent_lc.set_eval_mode(True)

    ################ Reset ################
    state, info = env.reset(
        _start_position_init=_hover_position,
        _target_position_init=_hover_position,
        _hover_position = _hover_position
    )
    # env.set_hover(_hover_pos)
    env.set_lc_target(_hover_position)

    ################ Hover loop ################
    t = 0
    done = False
    while not done and t < max_steps:

        agent_pos = env.agent_pos
        state_14 = env.state_14.copy()

        # LC error state
        state_14[3] = _hover_position[0] - agent_pos[0]
        state_14[4] = _hover_position[1] - agent_pos[1]
        low_state = state_14[:6]

        low_action = ppo_agent_lc.select_action(low_state)
        state, reward, terminated, truncated, info = env.step(low_action)
        t += 1

    env.close()
    print("Hover steps:", t)

########################### Mode 6: hovering task ###########################
def test_traj_only(_id_in=1):
    print("============================================================================================")
    print(f"========== TRAJECTORY ONLY TEST, seed={_id_in} ==========")

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
    true_random = True
    random_seed = 0 if true_random else _id_in
    _is_head = False
    max_ll_steps = 5
    #################################################

    ##################################################### Task Selection #####################################################
    flow_seed = _id_in
    test_mode = True
    theta_mode = False
    _is_normalize = True
    _is_near = False
    _fov = np.pi
    switch_mode = False
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    random_range = 32
    state_dim_mode = 6
    switch_range = 8
    force_clip = force_range = 15.0
    _is_avoid = False

    ################ Basic settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, _ = parser_1.parse_known_args()
    args_1.action_interval = 10

    ################ Env settings ################
    max_steps = 1500
    video_dir = "./0201_EXP_Trajectory/video_frames"

    # ===== Trajectory definition =====

    def generate_figure8_from_fixed_point( fixed_point=(180, 64), a=50, b=20, n_points=400):
        """
        Generate a closed figure-8 trajectory that
        starts and ends at the same fixed point.

        Returns:
            traj_points: (n_points, 2)
        """
        t = np.linspace(0, 2 * np.pi, n_points)

        x = a * np.sin(t)
        y = b * np.sin(2 * t)

        traj = np.vstack((x, y)).T
        traj += np.array(fixed_point)

        return traj
    
    def generate_circle_starting_at_point(start_point=(180, 64), radius=32, n_points=400):
        """
        Generate a closed circular trajectory that
        starts and ends at the given start point.
        """
        t = np.linspace(0, 2 * np.pi, n_points)

        x = radius * np.cos(t)
        y = radius * np.sin(t)

        # shift so that the first point is start_point
        traj = np.vstack((x, y)).T
        traj -= traj[0]
        traj += np.array(start_point)

        return traj
    
    def generate_ellipse_starting_at_right(start_point=(240, 64), a=50, b=20, n_points=400):
        """
        Generate a closed elliptical trajectory that
        starts at the rightmost point of the ellipse (major axis),
        and goes counter-clockwise.

        Parameters
        ----------
        start_point : tuple
            Center of the ellipse (xc, yc)
        a : float
            Semi-major axis length (horizontal)
        b : float
            Semi-minor axis length (vertical)
        n_points : int
            Number of points in the trajectory

        Returns
        -------
        traj_points : np.ndarray, shape (n_points, 2)
        """
        t = np.linspace(0, 2 * np.pi, n_points)

        # Standard parametric ellipse
        x = a * np.cos(t)
        y = b * np.sin(t)

        # Shift so first point is at rightmost vertex
        # Rightmost vertex is at x = a, y = 0
        traj = np.vstack((x, y)).T

        # Calculate offset to make first point exactly at start_point
        offset = np.array(start_point) - traj[0]
        traj += offset

        return traj
    
    def generate_two_half_circles_fixed():
        """
        Generate a trajectory composed of:
        1️⃣ Right half-circle from (240,64) around center (210,64) lower half to (180,64)
        2️⃣ Left half-circle from (180,64) around center (150,64) upper half to (120,64)
        """
        n_points = 200  # 每个半圆点数

        # -------- 第一个半圆：右侧出发，下半圆 --------
        center1 = np.array([210, 64])
        radius1 = 30  # 240-210
        # 下半圆顺时针
        theta1 = np.linspace(0, -np.pi, n_points)  # 0 对应右侧，-pi 对应左侧
        x1 = center1[0] + radius1 * np.cos(theta1)
        y1 = center1[1] + radius1 * np.sin(theta1)

        # -------- 第二个半圆：左侧出发，上半圆 --------
        center2 = np.array([150, 64])
        radius2 = 30  # 180-150
        # 上半圆逆时针
        theta2 = np.linspace(0, np.pi, n_points)  # 0 对应右侧，pi 对应左侧
        x2 = center2[0] + radius2 * np.cos(theta2)
        y2 = center2[1] + radius2 * np.sin(theta2)

        # 合并轨迹
        traj = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))

        return traj

    fixed_start_points = (180, 64)
    traj_points = generate_figure8_from_fixed_point(fixed_point=fixed_start_points, a=50, b=20, n_points=400)
    # traj_points = generate_circle_starting_at_point(start_point=fixed_start_points, radius=32, n_points=400)
    # fixed_start_points = (240, 64)
    # traj_points = generate_ellipse_starting_at_right(start_point=fixed_start_points, a=40, b=20, n_points=400)
    # traj_points = generate_two_half_circles_fixed()

    env = train_env.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=None, target_position=None, _include_flow=True,
        _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test=test_mode, _state_dim=state_dim_mode,
        _is_random=true_random, _set_random=random_seed, _switch_range=switch_range, u_range=force_range, v_range=force_range, _forward_only=_is_head, _obstacle_avoid=_is_avoid,
        _theta_mode=theta_mode, _is_switch=switch_mode, u_clip=force_clip, v_clip=force_clip, _test_info=False, video_dir=video_dir, _near_mode=_is_near, _fov=_fov, 
        custom_trajectory = traj_points, plot_mode="full", _plot_task = "tracking")

    ################ LC PPO ################
    lc_state_dim = 6
    ppo_agent_lc = Classic_PPO(lc_state_dim, env.lc_action_space.shape[0], lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
        action_std_init=action_std_4_test, continuous_action_output_scale=(env.lc_action_space.high - env.lc_action_space.low) / 2.0, continuous_action_output_bias=(env.lc_action_space.high + env.lc_action_space.low) / 2.0)
    ppo_agent_lc.load_full("./models/exp/lc_dim6.pth")
    ppo_agent_lc.set_eval_mode(True)

    ################ Reset ################
    state, info = env.reset(
        _start_position_init=traj_points[0] ,
        _target_position_init=traj_points[1] 
    )

    ################ Trajectory tracking loop ################
    t = 0
    done = False
    phase_step = 1
    traj_reach_th = 1.0
    env.set_traj(traj_points)
    agent_position = env.agent_pos.copy()

    while not done and t < max_steps:

        target_position = env.target_position
        lc_target_x, lc_target_y = target_position
        env.set_lc_target([lc_target_x, lc_target_y])

        state_14 = env.state_14.copy()
        state_14[3] = lc_target_x - agent_position[0]
        state_14[4] = lc_target_y - agent_position[1]
        low_state = state_14[:6]

        low_action = ppo_agent_lc.select_action(low_state)
        state, reward, terminated, truncated, info = env.step(low_action)
        agent_position = env.agent_pos.copy()
        t += 1

        if terminated:
            phase_step += 1
            if phase_step < len(traj_points):
                env.set_target(traj_points[phase_step])
            else:
                done = True
                print(colored(f"All targets reached at step {t_step}", "green"))

    env.close()
    print("Trajectory steps:", t)

########################### Mode 7: RL_test ###########################
def test_RL(_id_in=1):
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
    video_dir = "./0201_Video_RL/video_frames"
    #####################################################  

    ##################################################### Tasks #####################################################  
    ########### With Obstacle ###########    
    random_range = 56
        
    _id = 3
    test_mode = False
    switch_range = 8
    flow_seed = _id_in
    switch_mode = False
    _is_normalize = True
    _fov = np.pi
    _start = np.array([220.0, 64.0])
    _target = np.array([64.0, 64.0])
    _start_position = None
    _target_position = None
    max_steps = 600
    max_ep_len = max_steps + 20
    state_dim_mode = 10
    input_max = 15.0
    clip_max = 15.0  
    ppo_path = f'./models/exp/rl/dim{state_dim_mode}/{_id}.pth'

    ######################################################
    env = train_env_basic_2_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, start_position=_start_position, target_position=_target_position,
    _include_flow=True, _plot_flow=True, _proccess_flow=False, video_dir=video_dir,
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
    ######################################################

    ########################### Test ###########################
    # 1 env reset
    x_start = 232.68
    y_start = 56.88
    x_target = 74.37
    y_target = 53.06
    # _extra_obs = False
    _extra_obs = True

    _start_position_init = np.array([x_start, y_start])
    _target_position_init = np.array([x_target, y_target])
    state, info = env.reset(_start_position_init=_start_position_init, _target_position_init=_target_position_init, _add_virtual_cylinders=_extra_obs, _custom_cylinders=[(160, 64), (160, 108), (160, 16)])
    print("position_reset:", env.agent_pos)

    for i in range(1, max_ep_len + 1):
        # 2 env step
        action = ppo_agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            break
    env.close()
###########################



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="0=test(), 1=test_1(), 2=success_rate_test(), 3=success_rate_test_RL()")
    parser.add_argument("--id", type=int, default=0, help="Random seed (0=random, otherwise fixed)")
    parser.add_argument("--control", type=str, choices=["HRL", "HEADING", "JOINT"], default="HRL",
                        help="Control mode for success_rate_test")
    parser.add_argument("--task", type=str, choices=["avoid", "free"], default="avoid",
                        help="Task mode for success_rate_test")
    args = parser.parse_args()

    if args.mode == 0:
        test(control_mode=args.control, _id_in=args.id, TASK_MODE=args.task)
    elif args.mode == 1:
        test_1(args.id)
    elif args.mode == 2:
        success_rate_test(control_mode=args.control, _id_in=args.id, TASK_MODE=args.task)
        # python main.py 2 --id 5 --control HEADING --task free
    elif args.mode == 3:
        success_rate_test_RL(args.id)
    elif args.mode == 4:
        test_m(control_mode=args.control, _id_in=args.id, TASK_MODE=args.task)
    elif args.mode == 5:
        test_hover_only(args.id)
    elif args.mode == 6:
        test_traj_only(args.id)
    elif args.mode == 7:
        test_RL(args.id)












