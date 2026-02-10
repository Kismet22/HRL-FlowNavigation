import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse
from env_new import train_env_basic_hovering
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import matplotlib.colors as mcolors


max_steps = 400
parser_1 = argparse.ArgumentParser()
args_1, unknown = parser_1.parse_known_args()
args_1.action_interval = 10
# _start = np.array([float(240), float(64)])
# _target = np.array([float(190), float(64)])
_start = np.array([float(220), float(64)])
_target = np.array([float(64), float(64)])
env = train_env_basic_hovering.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _random_range=56)


# 全局变量存储 colorbar
colorbar = None

def plot_env(
    ax,
    target,
    circles,
    agent_pos,
    history,
    agent_angle,
    flow_x=None,
    flow_y=None,
    force=None,
    x_range=330,
    y_range=130
):
    ax.clear()

    # =========================
    # 背景：涡度场
    # =========================
    if flow_x is not None and flow_y is not None:
        dx = x_range / flow_x.shape[0]
        dy = y_range / flow_y.shape[1]

        dvdx = np.gradient(flow_y, dx, axis=0)
        dudy = np.gradient(flow_x, dy, axis=1)
        vorticity = dvdx - dudy

        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

        x = np.linspace(0, x_range, flow_x.shape[0])
        y = np.linspace(0, y_range, flow_y.shape[1])
        X, Y = np.meshgrid(x, y)

        ax.pcolormesh(
            X, Y,
            vorticity.T,
            cmap='seismic',
            norm=norm,
            shading='auto',
            alpha=0.7,
            zorder=1
        )

    # =========================
    # 目标点 + 目标半径
    # =========================
    ax.scatter(*target, color='magenta', zorder=5)

    # ax.add_patch(plt.Circle(
    #     target,
    #     16,
    #     edgecolor='red',
    #     linestyle='--',
    #     linewidth=2.5,
    #     fill=False,
    #     zorder=4
    # ))

    # =========================
    # 障碍物
    # =========================
    for circle in circles:
        ax.add_patch(plt.Circle(
            circle["center"],
            circle["radius"],
            color='black',
            alpha=0.3,
            zorder=4
        ))

    # =========================
    # 轨迹
    # =========================
    if history:
        hx, hy = zip(*history)
        ax.plot(
            hx, hy,
            linestyle='-',
            color='darkgreen',
            linewidth=2.5,
            zorder=4
        )

    # =========================
    # 智能体（椭圆）
    # =========================
    ellipse_h = circles[0]["radius"]
    ellipse_w = ellipse_h / 1.5

    ellipse = Ellipse(
        xy=agent_pos,
        width=ellipse_h,
        height=ellipse_w,
        angle=np.degrees(agent_angle),
        color='darkgreen',
        alpha=0.7,
        zorder=6
    )
    ax.add_patch(ellipse)

    # =========================
    # 合力方向（关键）
    # =========================
    if force is not None:
        Fx, Fy = force
        F_norm = np.linalg.norm(force) + 1e-8

        arrow_len = 20.0
        Fx_n = Fx / F_norm * arrow_len
        Fy_n = Fy / F_norm * arrow_len

        ax.quiver(
            agent_pos[0], agent_pos[1],
            Fx_n, Fy_n,
            angles='xy',
            scale_units='xy',
            scale=1,
            color='orange',
            width=0.004,
            headwidth=4,
            zorder=7
        )

    # =========================
    # 坐标轴（物理单位）
    # =========================
    scale = 16
    ax.set_xlim(0, x_range)
    ax.set_ylim(0, y_range)
    ax.set_aspect('equal', adjustable='box')

    xticks = np.arange(0, x_range + 1, 32)
    yticks = np.arange(0, y_range + 1, 32)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x/scale:.0f}" for x in xticks])
    ax.set_yticklabels([f"{y/scale:.0f}" for y in yticks])

    ax.grid(False)


# 重置函数：重置环境和历史轨迹
def reset_env():
    global history, agent_pos

    env.reset()

    # === 环境状态 ===
    target = env.target_position
    agent_pos = env.agent_pos
    agent_angle = env.state[5]
    circles = env.circles

    # === 力 & 流场 ===
    force = env.action[:2]        # 合力方向（Fx, Fy）
    flow_x = env.u_flow
    flow_y = env.v_flow

    # === 轨迹 ===
    history = [agent_pos]

    # === 绘图 ===
    plot_env(
        ax=ax,
        target=target,
        circles=circles,
        agent_pos=agent_pos,
        history=history,
        agent_angle=agent_angle,
        flow_x=flow_x,
        flow_y=flow_y,
        force=force,
        x_range=env.x_range,
        y_range=env.y_range
    )

    status_label.config(text="Status: Ready", foreground="green")
    canvas.draw()


# Tkinter 界面
def update_trajectory():
    try:
        # === 读取控制输入 ===
        action = [
            float(entry_x.get()),
            float(entry_y.get()),
            float(entry_z.get())
        ]

        _, _, terminated, truncated, _ = env.step(action)

        # === 当前环境状态 ===
        target = env.target_position
        circles = env.circles

        agent_pos = env.agent_pos
        agent_angle = env.state[5]

        # 合力（或控制力）
        force = env.action[:2]
        print(f"Force_x:{force[0]}; Force_y:{force[1]}")

        # 流场
        flow_x = env.u_flow
        flow_y = env.v_flow

        # 轨迹
        history.append(agent_pos)

        # === 绘图 ===
        plot_env(
            ax=ax,
            target=target,
            circles=circles,
            agent_pos=agent_pos,
            history=history,
            agent_angle=agent_angle,
            flow_x=flow_x,
            flow_y=flow_y,
            force=force,
            x_range=env.x_range,
            y_range=env.y_range
        )

        # === 状态提示 ===
        if terminated:
            status_label.config(text="Status: Terminated", foreground="red")
        else:
            status_label.config(text="Status: Running", foreground="green")

        canvas.draw()

    except ValueError:
        status_label.config(
            text="Invalid input. Please enter numeric values.",
            foreground="red"
        )





# ===============================
# 创建 Tkinter 主窗口
# ===============================
root = tk.Tk()
root.title("Trajectory Controller")

# ===============================
# 环境初始化
# ===============================
env.reset()

target = env.target_position
agent_pos = env.agent_pos
agent_angle = env.state[5]

flow_x = env.u_flow
flow_y = env.v_flow
circles = env.circles

# 初始合力（通常为 0 或 action[:2]）
force = env.action[:2]

# 轨迹历史
history = [agent_pos]

# ===============================
# 创建 matplotlib 图
# ===============================
fig, ax = plt.subplots(figsize=(16, 9))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, columnspan=4)

# 初始绘制
plot_env(
    ax=ax,
    target=target,
    circles=circles,
    agent_pos=agent_pos,
    history=history,
    agent_angle=agent_angle,
    flow_x=flow_x,
    flow_y=flow_y,
    force=force,
    x_range=env.x_range,
    y_range=env.y_range
)

canvas.draw()

# ===============================
# 控制面板
# ===============================
control_frame = tk.Frame(root)
control_frame.grid(row=1, column=0, columnspan=4, pady=10)

# === 输入框 ===
tk.Label(control_frame, text="a_x:").grid(row=0, column=0, padx=5)
entry_x = ttk.Entry(control_frame, width=10)
entry_x.grid(row=0, column=1, padx=5)

tk.Label(control_frame, text="a_y:").grid(row=0, column=2, padx=5)
entry_y = ttk.Entry(control_frame, width=10)
entry_y.grid(row=0, column=3, padx=5)

tk.Label(control_frame, text="a_w:").grid(row=0, column=4, padx=5)
entry_z = ttk.Entry(control_frame, width=10)
entry_z.grid(row=0, column=5, padx=5)

# === 更新按钮 ===
update_button = ttk.Button(
    control_frame,
    text="Update",
    command=update_trajectory
)
update_button.grid(row=0, column=6, padx=10)

# === 重置按钮 ===
reset_button = ttk.Button(
    control_frame,
    text="Reset",
    command=reset_env
)
reset_button.grid(row=0, column=7, padx=10)

# ===============================
# 状态显示
# ===============================
status_label = tk.Label(
    root,
    text="Status: Ready",
    foreground="green"
)
status_label.grid(row=2, column=0, columnspan=4, pady=10)

# ===============================
# 启动 Tkinter 主循环
# ===============================
root.mainloop()
