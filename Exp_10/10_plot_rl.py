# import os
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# import re

# # ----------------------------- 设置 -----------------------------
# folders = {
#     "Flow Blind": "./10_rl_traj/dim_3_test_results",
#     "Pressure-aware": "./10_rl_traj/dim_11_test_results",
#     "Velocity-aware": "./10_rl_traj/dim_6_test_results"
# }

# save_fig_path = "./trajectory_plots/all_conditions_top3.png"
# os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)

# # 更亮的轨迹颜色
# colors_condition = ["tomato", "deepskyblue", "limegreen"]
# success_radius = 4  # 成功判定范围

# # ----------------------------- 工具函数 -----------------------------
# def parse_position(text):
#     """从 summary 文件中解析形如 [x y] 的坐标"""
#     nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
#     return np.array([float(nums[0]), float(nums[1])]) if len(nums) >= 2 else np.array([0, 0])

# # ----------------------------- 绘图 -----------------------------
# plt.figure(figsize=(10, 7))
# ax = plt.gca()

# # 起点和终点颜色循环
# point_colors = ["orchid", "gold", "cyan", "magenta", "lime"]

# for dim_idx, (dim_name, folder) in enumerate(folders.items()):
#     traj_files = sorted([f for f in os.listdir(folder) if f.startswith("trajectory_test") and f.endswith(".csv")])
#     if not traj_files:
#         print(f"No trajectory files in {folder}")
#         continue

#     for i, traj_file in enumerate(traj_files[:3]):  # 只取前三条轨迹
#         traj_path = os.path.join(folder, traj_file)
#         traj_data = []
#         with open(traj_path, "r") as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 traj_data.append(row)
#         if not traj_data:
#             continue

#         # --- 轨迹数据 ---
#         traj_array = np.array([[float(r["x"]), float(r["y"])] for r in traj_data])
#         result = traj_data[0].get("result", "Unknown")

#         # --- 读取 summary 获取真实目标 ---
#         test_id = int(re.findall(r'\d+', traj_file)[0])
#         summary_path = os.path.join(folder, f"summary_test{test_id}.txt")
#         start_pos = target_pos = None
#         with open(summary_path, "r") as f:
#             for line in f:
#                 if "Start Position" in line:
#                     start_pos = parse_position(line)
#                 elif "Target Position" in line:
#                     target_pos = parse_position(line)

#         if target_pos is None:
#             print(f"Warning: no target info for {traj_file}")
#             continue

#         # --- 绘制轨迹（全部虚线，更亮的颜色） ---
#         color = colors_condition[dim_idx]
#         ax.plot(traj_array[:, 0], traj_array[:, 1],
#                 color=color, linestyle='-', linewidth=2.5, alpha=0.9,
#                 label=f"{dim_name}" if i == 0 else "")

#         # --- 起点和目标点每条轨迹不同颜色 ---
#         point_color = point_colors[i % len(point_colors)]

#         # 起点
#         if start_pos is not None:
#             ax.scatter(start_pos[0], start_pos[1],
#                        color=point_color, marker="o", s=50, edgecolor="black", zorder=5)

#         # 目标点及判定圈
#         ax.scatter(target_pos[0], target_pos[1],
#                    color=point_color, marker="*", s=90, edgecolor="black", zorder=6)
#         circle = plt.Circle(target_pos, success_radius,
#                             color=point_color, fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
#         ax.add_patch(circle)

# # ----------------------------- 美化 -----------------------------
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# # ax.set_title("Top 3 Trajectories Across Conditions (Dashed & Bright Colors)")
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# ax.set_aspect('equal', adjustable='box')

# plt.tight_layout()
# plt.savefig(save_fig_path, bbox_inches='tight', dpi=200)
# plt.close()

# print(f"✅ Saved bright dashed trajectory plot: {save_fig_path}")



import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D  # ✅ 用于手动创建图例项

# ----------------------------- 设置 -----------------------------
dim_name = "Velocity-aware (dim6)"
folder = "./10_rl_traj/dim_6_test_results"
save_fig_path = "./trajectory_plots/dim6_fail_with_circles_colored.png"
os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)

# 轨迹与点样式
traj_color = "green"       # ✅ 轨迹绿色
point_color = "orange"

# 三个判定圆半径与颜色
success_radii = [4, 6, 8]
radius_colors = ["red", "gold", "blue"]  # ✅ 不同半径不同颜色

# ----------------------------- 工具函数 -----------------------------
def parse_position(text):
    """从 summary 文件中解析形如 [x y] 的坐标"""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return np.array([float(nums[0]), float(nums[1])]) if len(nums) >= 2 else np.array([0, 0])

# ----------------------------- 绘图 -----------------------------
plt.figure(figsize=(8, 6))
ax = plt.gca()

traj_files = sorted([f for f in os.listdir(folder) if f.startswith("trajectory_test") and f.endswith(".csv")])
if not traj_files:
    print(f"No trajectory files in {folder}")
else:
    for traj_file in traj_files:
        traj_path = os.path.join(folder, traj_file)
        traj_data = []
        with open(traj_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                traj_data.append(row)
        if not traj_data:
            continue

        result = traj_data[0].get("result", "Unknown")
        if result != "Failure":  # 只绘制失败轨迹
            continue

        # --- 轨迹点 ---
        traj_array = np.array([[float(r["x"]), float(r["y"])] for r in traj_data])

        # --- 读取 summary 文件中的起点与终点 ---
        test_id = int(re.findall(r'\d+', traj_file)[0])
        summary_path = os.path.join(folder, f"summary_test{test_id}.txt")
        start_pos = target_pos = None
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                for line in f:
                    if "Start Position" in line:
                        start_pos = parse_position(line)
                    elif "Target Position" in line:
                        target_pos = parse_position(line)

        # --- 绘制轨迹（绿色） ---
        ax.plot(traj_array[:, 0], traj_array[:, 1],
                color=traj_color, linestyle='-', linewidth=2.2, alpha=0.9)

        # --- 起点 ---
        if start_pos is not None:
            ax.scatter(start_pos[0], start_pos[1],
                       color="green", marker="o", s=50, edgecolor="black", zorder=5)

        # --- 目标点与判定圆 ---
        if target_pos is not None:
            ax.scatter(target_pos[0], target_pos[1],
                       color=point_color, marker="*", s=90, edgecolor="black", zorder=6)

            # 绘制三个不同半径的圈
            for radius, r_color in zip(success_radii, radius_colors):
                circle = Circle(target_pos, radius,
                                color=r_color, fill=False,
                                linestyle='--', linewidth=1.5, alpha=0.8)
                ax.add_patch(circle)

# ----------------------------- 图例构建 -----------------------------
# ✅ 手动添加图例项（用虚线Line2D代替Circle）
legend_lines = [
    Line2D([0], [0], color=radius_colors[0], ls='-', lw=1.8, label=f"D = {success_radii[0]}"),
    Line2D([0], [0], color=radius_colors[1], ls='-', lw=1.8, label=f"D = {success_radii[1]}"),
    Line2D([0], [0], color=radius_colors[2], ls='-', lw=1.8, label=f"D = {success_radii[2]}"),
]

ax.legend(handles=legend_lines,
          loc='upper left', bbox_to_anchor=(1.02, 1.0))

# ----------------------------- 美化 -----------------------------
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(save_fig_path, bbox_inches='tight', dpi=200)
plt.close()

print(f"✅ Saved fixed legend trajectory plot for {dim_name} at {save_fig_path}")



