import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from termcolor import colored

# ----------------------------
# 配置
# ----------------------------
save_dir = "./hover_test_results"
controller_file = os.path.join(save_dir, "hover_results.pkl")
rl_file = os.path.join(save_dir, "hover_results_rl.pkl")
hrl_file = os.path.join(save_dir, "hover_results_planner.pkl")

min_required_steps = 200   # 筛选阈值
max_ep_len = 200           # 绘图最大步数
# ----------------------------
# 工具函数：计算均值曲线
# ----------------------------
def compute_mean_curves(results_all):
    grouped = defaultdict(list)
    max_len = 0

    for traj_data in results_all:
        r = traj_data["r"]
        angle = traj_data["angle"]
        hover_point = np.array(traj_data["hover_point"])
        traj = traj_data["trajectory"]

        key = (r, angle)
        if len(traj) < min_required_steps:
            continue

        dists = [np.linalg.norm(np.array(step["position"]) - hover_point) for step in traj]
        dists = np.array(dists)
        max_len = max(max_len, len(dists))
        grouped[key].append(dists)

    mean_curves = {}
    for key, dists_list in grouped.items():
        padded = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in dists_list]
        arr = np.vstack(padded)
        mean_curve = np.nanmean(arr, axis=0)
        mean_curves[key] = mean_curve

    return mean_curves

# ----------------------------
# 加载数据
# ----------------------------
def load_results(path):
    with open(path, "rb") as f:
        results = pickle.load(f)
    print(colored(f"Loaded {len(results)} trajectories from {path}", "green"))
    return results

results_controller = load_results(controller_file)
results_rl = load_results(rl_file)
results_hrl = load_results(hrl_file)

# ----------------------------
# 计算均值曲线
# ----------------------------
mean_controller = compute_mean_curves(results_controller)
mean_rl = compute_mean_curves(results_rl)
mean_hrl = compute_mean_curves(results_hrl)

# 获取 r 和 angle 集合
r_values = sorted(set(k[0] for k in mean_controller.keys()))
angle_values = sorted(set(k[1] for k in mean_controller.keys()))

# ----------------------------
# 绘制 Controller 图
# ----------------------------
fig, axes = plt.subplots(len(r_values), len(angle_values), figsize=(14, 10), sharex=True, squeeze=False)

for i, r in enumerate(r_values):
    # 本行 y 轴最大值
    row_max = 0
    for j, angle in enumerate(angle_values):
        key = (r, angle)
        if key in mean_controller:
            row_max = max(row_max, np.nanmax(mean_controller[key]))

    for j, angle in enumerate(angle_values):
        ax = axes[i, j]
        key = (r, angle)
        if key not in mean_controller:
            ax.axis("off")
            continue
        mean_curve = mean_controller[key]
        t_axis = np.arange(len(mean_curve))
        ax.plot(t_axis, mean_curve, color="royalblue", lw=2)
        ax.set_ylim(0, row_max * 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(f"r={r}, θ={angle}°", fontsize=10)
        if j == 0:
            ax.set_ylabel("Distance to Target")
        if i == len(r_values) - 1:
            ax.set_xlabel("Time Steps")

plt.tight_layout()
save_path = os.path.join(save_dir, "hover_controller_mean.png")
plt.savefig(save_path, dpi=300)
print(colored(f"✅ Saved Controller figure: {save_path}", "yellow"))
plt.close()

# ----------------------------
# 绘制 RL vs HRL 对比图
# ----------------------------
fig, axes = plt.subplots(len(r_values), len(angle_values), figsize=(14, 10), sharex=True, sharey=True, squeeze=False)

for i, r in enumerate(r_values):
    for j, angle in enumerate(angle_values):
        ax = axes[i, j]
        key = (r, angle)
        has_data = False
        if key in mean_rl:
            ax.plot(np.arange(len(mean_rl[key])), mean_rl[key], color="orange", lw=2, ls="--", label="RL")
            has_data = True
        if key in mean_hrl:
            ax.plot(np.arange(len(mean_hrl[key])), mean_hrl[key], color="green", lw=2, ls="-", label="HRL")
            has_data = True
        if not has_data:
            ax.axis("off")
            continue
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(f"r={r}, θ={angle}°", fontsize=10)
        if j == 0:
            ax.set_ylabel("Distance to Target")
        if i == len(r_values) - 1:
            ax.set_xlabel("Time Steps")

# 统一图例放右上角，不遮挡
handles, labels = [], []
for ax in axes.flatten():
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
if handles:
    fig.legend(handles[:2], labels[:2], loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=12)

plt.tight_layout()
save_path = os.path.join(save_dir, "hover_rl_vs_hrl_mean.png")
plt.savefig(save_path, dpi=300)
print(colored(f"✅ Saved RL vs HRL figure: {save_path}", "yellow"))
plt.close()
