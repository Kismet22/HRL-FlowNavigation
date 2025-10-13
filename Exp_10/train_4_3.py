import gym
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from termcolor import colored
import time

####### PPO_METHOD #######
import icm_ppo
from icm_ppo import ICM_PPO
import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO
import hjbppo
from hjbppo import HJB_PPO

####### ENVIRONMENT #######
from env_new import train_env_upper_2_3

####### DEVICE #######
device = my_ppo_net_1.device
# device = hjbppo.device
# device = icm_ppo.device


# def compute_relative_subgoal(angle, r):
#     angle_1 = angle + np.pi/2
#     dx = r * np.cos(angle_1)
#     dy = r * np.sin(angle_1)
#     return dx.item(), dy.item()

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


def train(model_id, render_mode=None, checkpoint_path_high=None, checkpoint_path_low=None):
    print("============================================================================================")

    ####### 训练环境变量设置 #######

    #########
    # state_dim_mode = 18
    state_dim_mode = 10
    # state_dim_mode = 16
    
    # state_dim_mode = 3
    # state_dim_mode = 6
    # state_dim_mode = 11
    # state_dim_mode = 14

    # lc_state_dim = 3
    lc_state_dim = 6
    # lc_state_dim = 11
    # lc_state_dim = 14
    #########

    # continuous action space; else discrete
    has_continuous_action_space = True

    # max time_steps in one episode
    max_steps = 300
    # max_steps = 210
    max_ep_len = max_steps + 20

    max_ll_steps = 3

    # break training loop if timesteps > max_training_timesteps
    hl_max_training_timesteps = int(6e5)
    max_training_timesteps = int(8e5)

    # Note : print/log frequencies should be more than max_ep_len
    #########
    # print avg reward in the interval (in num timesteps)
    print_freq = 1500
    #########

    #########
    # log avg reward in the interval (in num timesteps)
    log_freq = 1500
    #########

    #########
    # saving frequent
    save_model_freq = int(6e3)
    #########


    # starting std for action distribution (Multivariate Normal)
    action_std = 0.5  
    # set std for action distribution when testing(low_level controller)
    action_std_4_test = 0.04
    # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    action_std_decay_rate = 0.02  
    # minimum action_std (stop decay after action_std <= min_action_std)
    min_action_std = 0.1

    #########  
    # action_std decay frequency (in num timesteps)
    action_std_decay_freq = int(3e4)
    #########

    #########
    # update policy every n timesteps
    update_timestep = int(3e3)
    #########   


    ################ RL Hyperparameters ################
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
    #####################################################  


    ################ Other Settings ################
    # training ID, change this to prevent overwriting weights in same env_name folder
    pretrained_model_ID = model_id

    # set flow seed if required (0 : random flow; else : fixed flow)
    flow_seed = 0
    # flow_seed = 1

    # set heading mode
    _is_head = False # True : forward only; else : all direction
    lc_train = False # True: train from scratch; False: use pre-trained policy
    theta_mode = False
    switch_mode = False
    _obstacle_mode = True
    _is_normalize = True
    obs_task = True
    

    # ppo_path = './models/easy_task/0705_lc.pth' # all direction
    ppo_path = f'./models/exp/lc_dim{lc_state_dim}.pth'

    # set random mode(True : real random mode; else : random.seed(random_seed))
    true_random = True
    if true_random:
        random_seed = 0
    else:
        random_seed = 1
    
    test_mode = True
    switch_range = 1
    # test_mode = False
    # switch_range = 8
    
    input_max = 15.0
    clip_max = 15.0

    # input_max = 20.0
    # clip_max = 20.0    
    #####################################################


    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    if _is_head:
        if obs_task:
            _start = np.array([float(220), float(64)])
            _target = np.array([float(64), float(64)])
            random_range = 56
        else:
            _start = np.array([float(240), float(64)])
            _target = np.array([float(160), float(64)])
            random_range = 32
        env_name = f'Flow_env_Upper_dim_{state_dim_mode}_lcforward'
    else:
        if obs_task:
            _start = np.array([float(220), float(64)])
            _target = np.array([float(64), float(64)])
            random_range = 56
        else:
            _start = np.array([float(240), float(64)])
            _target = np.array([float(160), float(64)])
            random_range = 32
        if theta_mode:
            env_name = f'Flow_env_Upper_dim_{state_dim_mode}_lcalldirection_actiondim1'
        else:
            if not _obstacle_mode:
                env_name = f'Flow_env_Upper_dim_{state_dim_mode}_lcalldirection_actiondim2'
            else:
                env_name = f'Flow_env_Upper_dim_{state_dim_mode}_obstacle_new'

    env = train_env_upper_2_3.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=input_max, v_range=input_max, 
    _forward_only=_is_head, _obstacle_avoid=True, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=clip_max, v_clip=clip_max, _obstacle_mode=_obstacle_mode)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # high-level controller action space
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    

    # low-level controller action space
    lc_action_output_scale = np.array([])
    lc_action_output_bias = np.array([])
    if has_continuous_action_space:
        # low-level controller action space dimension
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n


    ###########################################################
    # -------------------- 日志与保存路径 --------------------
    ###########################################################
    # PPO 日志文件夹
    log_dir = os.path.join("PPO_logs", env_name)
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名
    rl_method = "PPO"
    log_f_name = os.path.join(
        log_dir,
        f"{rl_method}_{env_name}_log_{pretrained_model_ID}_seed_{random_seed}" + ("_normalize.csv" if _is_normalize else ".csv")
    )

    print(f"Current logging model ID for {env_name}: {pretrained_model_ID}")
    print(f"Logging at: {log_f_name}")

    # 检查日志文件是否存在，并读取最后一行
    i_episode, time_step, max_model_avg_return = 0, 0, -np.inf
    # High-Level Steps
    hl_time_step = 0  
    if os.path.exists(log_f_name):
        print(f"Warning: Log file already exists. Appending new data.")
        try:
            with open(log_f_name, "r") as log_f:
                lines = log_f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    print("Last line of log file:")
                    print(last_line)
                    try:
                        last_line_values = last_line.split(',')
                        i_episode = int(last_line_values[0])
                        time_step = int(last_line_values[1])
                        hl_time_step = int(last_line_values[2])
                        max_model_avg_return = float(last_line_values[3])
                        action_std = float(last_line_values[4])
                        print(f"Loaded from last log entry: i_episode={i_episode}, "
                            f"time_step={time_step}, high_steps={hl_time_step}, max_model_avg_return={max_model_avg_return}, std={action_std}")
                    except ValueError:
                        print("Failed to parse last line. Starting from 0.")
        except Exception as e:
            print(f"Failed to read log file: {e}")
    else:
        # 创建新日志文件并写入表头
        try:
            with open(log_f_name, "w") as log_f:
                log_f.write('episode,timestep,high_steps,avg_return_in_period,action_std,ratio,avg_return_in_period_lc\n')
                print("Log file created and header written.")
        except Exception as e:
            print(f"Failed to create log file: {e}")

    # PPO checkpoint 文件夹
    checkpoint_dir = os.path.join("PPO_preTrained", env_name, f"PPO_{env_name}_ID_{pretrained_model_ID}_seed_{random_seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint save directory: {checkpoint_dir}")
    directory = checkpoint_dir

    ###########################################################
    # -------------------- 打印训练信息 -----------------------
    ###########################################################

    print("="*80)
    print(f"Max training timesteps: {max_training_timesteps}")
    print(f"Max timesteps per episode: {max_ep_len}")
    print(f"Model saving frequency: {save_model_freq} timesteps")
    print(f"Log frequency: {log_freq} timesteps")
    print(f"Printing average reward over last: {print_freq} timesteps")
    print("-"*80)
    print(f"State space dimension: {state_dim}")
    print(f"Action space dimension: {action_dim}")
    print("-"*80)
    if has_continuous_action_space:
        print("Continuous action space policy")
        print(f"Starting std: {action_std}, Decay rate: {action_std_decay_rate}, Min std: {min_action_std}, Decay freq: {action_std_decay_freq}")
    else:
        print("Discrete action space policy")
    print("-"*80)
    print(f"RL update frequency: {update_timestep} timesteps, K epochs: {K_epochs}, Epsilon clip: {eps_clip}, Gamma: {gamma}")
    print(f"Actor LR: {lr_actor}, Critic LR: {lr_critic}")
    if random_seed:
        print(f"Random seed: {random_seed}")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    print("="*80)

    ###########################################################
    # -------------------- PPO Agent 初始化 -------------------
    ###########################################################

    # 高层 PPO
    ppo_agent = Classic_PPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std, continuous_action_output_scale=action_output_scale,
        continuous_action_output_bias=action_output_bias
    )

    # 低层 PPO
    if lc_train:
        print("[Low-Level Controller] Training from scratch.")
        ppo_agent_lc = Classic_PPO(
            lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
            has_continuous_action_space,
            action_std,
            continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias
        )
    else:
        print(f"[Low-Level Controller] Loading pre-trained model from: {ppo_path}")
        ppo_agent_lc = Classic_PPO(
            lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
            has_continuous_action_space,
            action_std_init=action_std_4_test,
            continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias
        )
        ppo_agent_lc.load_full(ppo_path)

    ###########################################################
    # -------------------- 可选加载 checkpoint -----------------
    ###########################################################

    if checkpoint_path_high:
        print(f"Loading high-level checkpoint from: {checkpoint_path_high}")
        ppo_agent.load_full(checkpoint_path_high)

    if checkpoint_path_low:
        print(f"Loading low-level checkpoint from: {checkpoint_path_low}")
        ppo_agent_lc.load_full(checkpoint_path_low)

    ################################################################################################################## 
    ################# Printing and Logging variables ################
    print_running_return = 0
    print_running_episodes = 0
    log_running_return = 0
    log_running_episodes = 0
    cur_model_running_return = 0
    cur_model_running_episodes = 0
    if lc_train:
        lc_print_running_return = 0
        lc_print_running_episodes = 0
        lc_log_running_return = 0
        lc_log_running_episodes = 0
        # lc_cur_model_running_return = 0
        # lc_cur_model_running_episodes = 0

    # write training data
    log_f = open(log_f_name, "a")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")
    ################################################################################################################## 

    ################################################################################################################## 
    ################# Training Procedure ################
    icm_ratio = 1
    subgoal_threshold = 1.0
    if _is_head:
        print(colored("**** Forward Mode **** ", 'green'))
    else:
        print(colored("**** Normal Mode **** ", 'green'))

    # while time_step <= max_training_timesteps:
    while hl_time_step <= hl_max_training_timesteps:
        # === 环境初始化 ===
        state, info = env.reset()
        state_14 = env.state_14
        agent_position = env.agent_pos
        target_position = env.target_position

        # 保存上一帧有效状态（用于差分/奖励）
        prev_state = state.copy()
        prev_state_14 = state_14.copy()
        prev_agent_position = agent_position.copy()

        print("position_reset:", agent_position)

        # === Episode 统计变量 ===
        current_ep_return = 0
        lc_current_ep_return = 0 if lc_train else None
        t = 0
        done = False

        # === 高层参考角度 ===
        target_2_agent = np.arctan2(
            target_position[1] - agent_position[1],
            target_position[0] - agent_position[0]
        )
        if _is_head:
            # prev_high_action = target_2_agent - np.pi/2
            # prev_high_action = np.pi/2
            prev_high_action = None
        else:
            prev_high_action = target_2_agent

        # === 高层控制循环 ===
        while not done and t < max_ep_len:
            # === 高层决策：每 max_ll_steps 步更新一次 ===
            high_state = state
            high_action = ppo_agent.select_action(high_state)     # 高层输出目标角度
            high_action = np.clip(high_action, env.low, env.high) # 限制动作范围
            hl_time_step += 1

            # === 将角度转化为子目标坐标 ===
            agent_position = env.agent_pos

            # lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=high_action[0], r=high_action[1])
            if _is_head:
                lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=high_action, r=2)
            else:
                if not theta_mode:
                    vec = high_action
                    norm = np.linalg.norm(vec) + 1e-6
                    vec = vec / norm
                    theta = np.arctan2(vec[1], vec[0])
                    r = 2
                else:
                    theta = high_action
                    r = 2
                lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=theta, r=r)
            relative_dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))

            cumulative_high_reward = 0
            ll_step = 0
            done_lc = False

            # === Smoothness penalty for high-level action difference ===
            if prev_high_action is not None:
                high_action_diff = np.linalg.norm(high_action - prev_high_action)
                smoothness_penalty = -20 * high_action_diff/np.pi
                cumulative_high_reward += smoothness_penalty
            prev_high_action = high_action.copy()


            # === 低层控制循环 ===
            lc_current_ep_return = 0
            while ll_step < max_ll_steps:
                agent_angle_old = env.angle

                # 构造低层状态：加上子目标相对位置
                low_state = state_14.copy()
                low_state[3] = lc_target_x - agent_position[0]
                low_state[4] = lc_target_y - agent_position[1]

                if lc_state_dim == 3:
                    low_state = low_state[3:6]
                elif lc_state_dim == 6:
                    low_state = low_state[:6]
                elif lc_state_dim == 11:
                    low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

                dist_to_subgoal_old = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))

                # === 执行低层动作 ===
                if np.isnan(low_state).any():
                    print(colored(f"[!!!!!!!!!! Warning !!!!!!!!!!] NaN detected in low_state before select_action, timestep {time_step}. Replacing with safe values.", 'red'))
                    if prev_state_14 is not None and not np.isnan(prev_state_14).any():
                        if lc_state_dim == 3:
                            low_state = prev_state_14[3:6].copy()
                        elif lc_state_dim == 6:
                            low_state = prev_state_14[:6].copy()
                        elif lc_state_dim == 11:
                            low_state = np.hstack([prev_state_14[3:6], prev_state_14[6:14]]).astype(np.float32)
                    else:
                        low_state = np.nan_to_num(low_state, nan=0.0, posinf=240, neginf=0.0).astype(np.float32)

                low_action = ppo_agent_lc.select_action(low_state)
                state, reward, terminated, truncated, info = env.step(low_action)
                agent_position = env.agent_pos
                state_14 = env.state_14

                # === NaN 检测与替换 ===
                if np.isnan(state).any() or np.isnan(low_state).any():
                    truncated = True
                    reward = -500

                # === 更新上一帧有效状态 ===
                prev_state = state.copy()
                prev_state_14 = state_14.copy()
                prev_agent_position = agent_position.copy()

                done = terminated or truncated

                dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
                # === 判断低层是否完成子目标、失败或到达最大步数 ===
                if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                    done_lc = True

                ##############################################################################
                # 低层训练
                if lc_train:
                    # === 低层奖励 ===
                    reward_lc = -0.05 + 10 * (dist_to_subgoal_old - dist_to_subgoal)
                    # === 角度惩罚（鼓励平滑）===
                    angle_diff = np.abs(np.arctan2(np.sin(env.angle - agent_angle_old),
                                np.cos(env.angle - agent_angle_old)))
                    reward_lc += -20 * angle_diff/(np.pi)
                    angle_abs = np.abs(env.angle)
                    if angle_abs > np.pi/2:
                        excess = angle_abs - np.pi/2
                        reward_lc += - 360 * min(excess, np.pi/36)/(np.pi)
                    if dist_to_subgoal < subgoal_threshold:
                        reward_lc += 50
                    # === 存储 ===
                    ppo_agent_lc.buffer.rewards.append(reward_lc)
                    ppo_agent_lc.buffer.is_terminals.append(done_lc)
                    lc_current_ep_return += reward_lc
                ##############################################################################
                
                # === 更新计数器 ===
                cumulative_high_reward += reward
                time_step += 1
                t += 1
                ll_step += 1
                if done_lc:
                    break
            
            if lc_train:
                lc_print_running_return += lc_current_ep_return
                lc_print_running_episodes += 1
                lc_log_running_return += lc_current_ep_return
                lc_log_running_episodes += 1
                # lc_cur_model_running_return += lc_current_ep_return
                # lc_cur_model_running_episodes += 1

            # === 混合训练：根据子目标的接近程度 ===
            if lc_train:
                cumulative_high_reward += 10 * max((relative_dist_to_subgoal - dist_to_subgoal), -2)
            current_ep_return += cumulative_high_reward

            # === 存储高层经验 ===
            ppo_agent.buffer.rewards.append(cumulative_high_reward)
            ppo_agent.buffer.is_terminals.append(done)

            # saving next_state
            # state_next = torch.tensor(state, device=device, dtype=torch.float32)
            # ppo_agent.buffer.next_states.append(state_next)

            # === Update ===
            if hl_time_step % update_timestep == 0:
                # return per episode
                cur_model_avg_return = cur_model_running_return / cur_model_running_episodes
                if cur_model_avg_return > max_model_avg_return:
                    print("Better Model.")
                    # checkpoint_path = directory + f"best_model_time_step{time_step}_std{ppo_agent.action_std}.pth"
                    checkpoint_path = os.path.join(directory, f"best_model_time_step{time_step}_std{ppo_agent.action_std}.pth")
                    if lc_train:
                        # checkpoint_path_lc = directory + f"best_model_time_step{time_step}_std{ppo_agent_lc.action_std}_lc.pth"
                        checkpoint_path_lc = os.path.join(directory, f"best_model_time_step{time_step}_std{ppo_agent_lc.action_std}_lc.pth")
                    print("saving model at : " + checkpoint_path)

                    ############################################################
                    ppo_agent.save_full(checkpoint_path)
                    if lc_train:
                        ppo_agent_lc.save_full(checkpoint_path_lc)
                    # ppo_agent.save_full_icm(checkpoint_path)
                    ############################################################

                    max_model_avg_return = cur_model_avg_return
                cur_model_running_return = 0
                cur_model_running_episodes = 0

                print("Policy Updated. At Episode : {}  Timestep : {}".format(i_episode, time_step))
                ppo_agent.update()
                if lc_train:
                    ppo_agent_lc.update()
                
                ############################################################
                # icm_ratio = ppo_agent.icm_average_ratio
                ############################################################

            # if continuous action space; then decay action std of output action distribution
            if has_continuous_action_space and hl_time_step % action_std_decay_freq == 0:
                # 每action_std_decay_freq步，降低一次网络的初始化方差
                action_std = ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                if lc_train:
                    _ = ppo_agent_lc.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file， 多个episode进行一次记录，记录的是这几个episode的avg return
            if hl_time_step % log_freq == 0:
                # log average return till last episode
                log_avg_return = log_running_return / log_running_episodes
                log_avg_return = round(log_avg_return, 4)
                if lc_train:
                    lc_log_avg_return = lc_log_running_return / lc_log_running_episodes
                    lc_log_avg_return = round(lc_log_avg_return, 4)
                else:
                    lc_log_avg_return = 0
                log_f.write('{},{},{},{},{},{},{}\n'.format(i_episode, time_step, hl_time_step, log_avg_return, action_std, icm_ratio, lc_log_avg_return))
                log_f.flush()
                log_running_return = 0
                log_running_episodes = 0
                if lc_train:
                    lc_log_running_return = 0
                    lc_log_running_episodes = 0

            # printing average reward
            if hl_time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_return = print_running_return / print_running_episodes
                print_avg_return = round(print_avg_return, 2)
                if lc_train:
                    lc_print_avg_return = lc_print_running_return / lc_print_running_episodes
                    lc_print_avg_return = round(lc_print_avg_return, 2)
                else:
                    lc_print_avg_return = 0

                print("Episode : {} \t\t Timestep : {} \t\t Average Return H : {} \t\t Average Return L : {}".format(i_episode, time_step, print_avg_return, lc_print_avg_return))

                print_running_return = 0
                print_running_episodes = 0
                if lc_train:
                    lc_print_running_return = 0
                    lc_print_running_episodes = 0

            # save model weights
            if hl_time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                # checkpoint_path = directory + f"timestep_{time_step}_std_{ppo_agent.action_std:.2f}.pth"
                checkpoint_path = os.path.join(directory, f"timestep_{time_step}_std_{ppo_agent.action_std:.2f}.pth")
                if lc_train:
                    # checkpoint_path_lc = directory + f"timestep_{time_step}_std_{ppo_agent_lc.action_std:.2f}_lc.pth"
                    checkpoint_path_lc = os.path.join(directory, f"timestep_{time_step}_std_{ppo_agent_lc.action_std:.2f}_lc.pth")
                print("saving model at : " + checkpoint_path)

                ############################################################
                ppo_agent.save_full(checkpoint_path)
                if lc_train:
                    ppo_agent_lc.save_full(checkpoint_path_lc)
                # ppo_agent.save_full_icm(checkpoint_path)
                ############################################################

                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                print(colored(f"**** Episode Reward:{current_ep_return} **** ", "yellow"))
                break
            if t == max_ep_len:
                print(colored("**** Episode Terminated **** Reaches Training Episode Time limit.", 'blue'))
        print_running_return += current_ep_return
        print_running_episodes += 1

        log_running_return += current_ep_return
        log_running_episodes += 1

        cur_model_running_return += current_ep_return
        cur_model_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.terminate()

    # print total_step training time
    print("============================================================================================")
    _end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", _end_time)
    print("Total training time  : ", _end_time - start_time)
    print("============================================================================================")
    


if __name__ == '__main__':
    import argparse
    import time
    from datetime import datetime
    from termcolor import colored

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Train hierarchical PPO")

    # 必选参数：开始ID和训练次数
    parser.add_argument('start_id', type=int, help='Starting ID for training')
    parser.add_argument('train_times', type=int, help='Number of training times')

    # 可选参数：分别指定高层和低层模型的checkpoint路径
    parser.add_argument('--checkpoint-path-high', type=str, default=None,
                        help='Path to high-level PPO checkpoint folder')
    parser.add_argument('--checkpoint-path-low', type=str, default=None,
                        help='Path to low-level PPO checkpoint folder')

    # 解析命令行参数
    args = parser.parse_args()

    start_id = args.start_id
    train_times = args.train_times
    checkpoint_path_high = args.checkpoint_path_high
    checkpoint_path_low = args.checkpoint_path_low

    # 训练延迟启动提示
    total_delay = 1  # 可以根据需要设置为更长，例如 10
    interval = total_delay if total_delay < 10 else 10

    for remaining in range(total_delay, 0, -interval):
        print(f"程序将在 {remaining} 秒后开始...")
        time.sleep(interval)

    print("开始执行程序...")

    start_time = datetime.now().replace(microsecond=0)

    # 训练循环
    for i in range(start_id, start_id + train_times):
        print(f'\n======================== Training ID = {i} ========================')
        train(
            model_id=i,
            checkpoint_path_high=checkpoint_path_high,
            checkpoint_path_low=checkpoint_path_low
        )

    end_time = datetime.now().replace(microsecond=0)

    # 输出训练总结
    print(colored("============================================================================================", 'red'))
    print(colored(f"Training_TEST for {train_times} times Ended.", 'red'))
    print(colored(f"Started training_test at (GMT) : {start_time}", 'red'))
    print(colored(f"Finished training_test at (GMT) : {end_time}", 'red'))
    print(colored(f"Total training_test time  : {end_time - start_time}", 'red'))
    print(colored("============================================================================================", 'red'))
    pass
