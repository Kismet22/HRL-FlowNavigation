import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from termcolor import colored
import matplotlib.patches as mpatches

# ============================================================
# 1. 加载轨迹数据
# ============================================================
def load_trajectories(save_paths):
    """
    save_paths: dict, {method_name: path_to_pickle}
    return: dict, {method_name: list_of_trajectories}
    """
    results_all = {}
    for method, path in save_paths.items():
        with open(path, "rb") as f:
            results_all[method] = pickle.load(f)
        print(colored(f"Loaded {len(results_all[method])} trajectories for {method}", "green"))
    return results_all

# ============================================================
# 2. 统计指标
# ============================================================
def compute_and_save_metrics(results_all, save_dir="./hover_test_results", min_steps=200):
    """
    计算轨迹指标、统计成功率，并保存两个 CSV：
        1. hover_metrics.csv: 每条轨迹指标
        2. hover_success_rate.csv: 按 r, angle 分组及总体成功率
    只统计成功轨迹（长度 >= min_steps）
    
    指标包括：
        - mean_distance, final_distance, max_distance, distance_var
        - mean_speed, mean_omega, mean_action_norm
    成功率统计：
        - step200 (>=200)
        - max32, max24, max16 (排除超远轨迹)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # -------------------------------
    # 初始化
    # -------------------------------
    metrics_all = []
    success_rate_dict = defaultdict(lambda: defaultdict(dict))  # method -> (r, angle) / "overall" -> dict
    
    # 统计总数
    total_counts = defaultdict(lambda: defaultdict(int))
    success_counts = defaultdict(lambda: defaultdict(lambda: {"step200":0,"max32":0,"max24":0,"max16":0}))
    
    for method, traj_list in results_all.items():
        overall_total = 0
        overall_step200 = 0
        overall_max32 = 0
        overall_max24 = 0
        overall_max16 = 0
        
        for traj_data in traj_list:
            r = traj_data["r"]
            angle = traj_data["angle"]
            hover_point = np.array(traj_data["hover_point"])
            traj = traj_data["trajectory"]
            key = (r, angle)
            
            overall_total += 1
            total_counts[method][key] += 1
            
            positions = np.array([step["position"] for step in traj])
            distances = np.linalg.norm(positions - hover_point, axis=1)
            max_distance = distances.max()
            
            # step200
            if len(traj) >= min_steps:
                overall_step200 += 1
                success_counts[method][key]["step200"] += 1
                
                if max_distance <= 32:
                    overall_max32 += 1
                    success_counts[method][key]["max32"] += 1
                if max_distance <= 24:
                    overall_max24 += 1
                    success_counts[method][key]["max24"] += 1
                if max_distance <= 16:
                    overall_max16 += 1
                    success_counts[method][key]["max16"] += 1

                # 计算指标
                mean_distance = distances.mean()
                final_distance = distances[-1]
                distance_var = distances.var()
                mean_speed = np.mean([step["speed"] for step in traj])
                mean_omega = np.mean([step["omega"] for step in traj])
                mean_action_norm = np.mean([np.linalg.norm(step["action"]) for step in traj])
                
                metrics_all.append({
                    "method": method,
                    "r": r,
                    "angle": angle,
                    "total_steps": len(traj),
                    "mean_distance": mean_distance,
                    "final_distance": final_distance,
                    "max_distance": max_distance,
                    "distance_var": distance_var,
                    "mean_speed": mean_speed,
                    "mean_omega": mean_omega,
                    "mean_action_norm": mean_action_norm
                })
        
        # 保存总体成功率
        success_rate_dict[method]["overall"] = {
            "step200": overall_step200 / overall_total if overall_total>0 else np.nan,
            "max32": overall_max32 / overall_total if overall_total>0 else np.nan,
            "max24": overall_max24 / overall_total if overall_total>0 else np.nan,
            "max16": overall_max16 / overall_total if overall_total>0 else np.nan
        }
    
    # 保存按 r, theta 分组成功率
    for method in success_counts:
        for key in success_counts[method]:
            total = total_counts[method][key]
            counts = success_counts[method][key]
            success_rate_dict[method][key] = {
                "step200": counts["step200"]/total if total>0 else np.nan,
                "max32": counts["max32"]/total if total>0 else np.nan,
                "max24": counts["max24"]/total if total>0 else np.nan,
                "max16": counts["max16"]/total if total>0 else np.nan
            }
    
    # -------------------------------
    # 保存指标 CSV
    # -------------------------------
    df_metrics = pd.DataFrame(metrics_all)
    metrics_path = os.path.join(save_dir, "hover_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"✅ Metrics CSV saved to {metrics_path}")
    
    # -------------------------------
    # 保存成功率 CSV
    # -------------------------------
    rows = []
    for method, data in success_rate_dict.items():
        # overall
        overall = data["overall"]
        rows.append({
            "method": method,
            "r": "all",
            "angle": "all",
            "step200": overall["step200"],
            "max32": overall["max32"],
            "max24": overall["max24"],
            "max16": overall["max16"]
        })
        # 按 r,angle
        for key, counts in data.items():
            if key == "overall":
                continue
            r, angle = key
            rows.append({
                "method": method,
                "r": r,
                "angle": angle,
                "step200": counts["step200"],
                "max32": counts["max32"],
                "max24": counts["max24"],
                "max16": counts["max16"]
            })
    df_success = pd.DataFrame(rows)
    success_path = os.path.join(save_dir, "hover_success_rate.csv")
    df_success.to_csv(success_path, index=False)
    print(f"✅ Success rate CSV saved to {success_path}")
    
    return df_metrics, df_success





# def compute_metrics(results_all, min_steps=200):
#     """
#     计算轨迹均值距离、最大距离、最终距离、速度、角速度、动作幅值等指标
#     只保留成功轨迹（长度 >= min_steps）
#     返回 DataFrame 和分组距离字典
#     """

#     metrics_all = []
#     grouped_distances = defaultdict(lambda: defaultdict(list))  # method -> (r, angle) -> list of mean distances

#     for method, traj_list in results_all.items():
#         for traj_data in traj_list:
#             r = traj_data["r"]
#             angle = traj_data["angle"]
#             hover_point = np.array(traj_data["hover_point"])
#             traj = traj_data["trajectory"]
#             key = (r, angle)

#             total_steps = len(traj)
#             success = total_steps >= min_steps
#             if not success:
#                 continue

#             positions = np.array([step["position"] for step in traj])
#             distances = np.linalg.norm(positions - hover_point, axis=1)

#             mean_distance = distances.mean()
#             final_distance = distances[-1]
#             max_distance = distances.max()
#             distance_var = distances.var()  # 新增距离方差
#             mean_speed = np.mean([step["speed"] for step in traj])
#             mean_omega = np.mean([step["omega"] for step in traj])
#             mean_action_norm = np.mean([np.linalg.norm(step["action"]) for step in traj])

#             metrics_all.append({
#                 "method": method,
#                 "r": r,
#                 "angle": angle,
#                 "total_steps": total_steps,
#                 "mean_distance": mean_distance,
#                 "final_distance": final_distance,
#                 "max_distance": max_distance,
#                 "distance_var": distance_var,  # 新增
#                 "mean_speed": mean_speed,
#                 "mean_omega": mean_omega,
#                 "mean_action_norm": mean_action_norm,
#             })

#             grouped_distances[method][key].append(distances)

#     df_metrics = pd.DataFrame(metrics_all)
#     return df_metrics, grouped_distances

def plot_hover_success_rate_by_r(csv_path, save_dir="./hover_test_results"):
    """
    根据 r 值绘制 hover success rate 曲线，并保存为多张图片。
    横轴为 step200, max32, max24, max16
    r=0 和 r='all' 单独一张图（不使用子图）
    其他 r 值每个 angle 一个子图，三列布局
    不同 method 用不同颜色区分
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 将 r 列统一转换为字符串，去掉空格
    df['r'] = df['r'].apply(lambda x: str(x).strip())
    
    # 横轴顺序
    x_labels = ['step200', 'max32', 'max24', 'max16']
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # r 的不同取值
    r_values = df['r'].unique()
    
    # 设置颜色
    colors = ['r', 'g', 'b', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    for r_val in r_values:
        subset = df[df['r'] == r_val]
        
        # r='all' 或 r='0' 单独一张图
        if r_val.lower() == 'all' or r_val == '0':
            plt.figure(figsize=(8, 6))
            title_prefix = "All Chases" if r_val.lower() == 'all' else "No initial offset"
            
            for angle_val in subset['angle'].unique():
                angle_subset = subset[subset['angle'] == angle_val]
                for i, method in enumerate(angle_subset['method'].unique()):
                    method_data = angle_subset[angle_subset['method'] == method]
                    y_values = method_data[x_labels].values.flatten()
                    plt.plot(x_labels, y_values, marker='o', color=colors[i % len(colors)],
                             label=f"{method}")
            
            plt.title(title_prefix)
            plt.xlabel("Test Cases")
            plt.ylabel("Success Rate")
            plt.ylim(0, 1.05)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            save_name = "hover_success_rate_all.png" if r_val.lower() == 'all' else "hover_success_rate_r0.png"
            save_path = os.path.join(save_dir, save_name)
            plt.savefig(save_path, dpi=300)
            print(f"保存图片: {save_path}")
            plt.close()
        
        else:
            # 多角度子图，三列布局
            angles = subset['angle'].unique()
            n_angles = len(angles)
            ncols = 3
            nrows = (n_angles + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
            axes = axes.flatten()
            
            for idx, angle_val in enumerate(angles):
                ax = axes[idx]
                angle_subset = subset[subset['angle'] == angle_val]
                
                for i, method in enumerate(angle_subset['method'].unique()):
                    method_data = angle_subset[angle_subset['method'] == method]
                    y_values = method_data[x_labels].values.flatten()
                    ax.plot(x_labels, y_values, marker='o', color=colors[i % len(colors)],
                            label=f"{method}")
                
                ax.set_title(f"r={r_val}, angle={angle_val}°")
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("Test Cases")
                ax.set_ylabel("Success Rate")
                ax.grid(True)
                ax.legend()
            
            # 隐藏多余子图
            for j in range(idx+1, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"hover_success_rate_r{r_val}.png")
            plt.savefig(save_path, dpi=300)
            print(f"保存图片: {save_path}")
            plt.close()


# ============================================================
# 3. 绘制均值距离对比图
# ============================================================
def plot_mean_distance(grouped_distances, methods, colors, r_values=None, angle_values=None, save_path=None):
    """
    绘制多方法均值距离对比图
    每个子图显示 r, θ
    整张图没有全局标题
    """
    if r_values is None:
        r_values = sorted({k[0] for m in grouped_distances.values() for k in m.keys()})
    if angle_values is None:
        angle_values = sorted({k[1] for m in grouped_distances.values() for k in m.keys()})

    fig, axes = plt.subplots(len(r_values), len(angle_values), figsize=(14, 10))
    
    # 处理 axes 统一格式
    if len(r_values) == 1 and len(angle_values) == 1:
        axes = np.array([[axes]])
    elif len(r_values) == 1:
        axes = np.array([axes])
    elif len(angle_values) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, r in enumerate(r_values):
        for j, angle in enumerate(angle_values):
            ax = axes[i, j]
            key = (r, angle)
            plotted = False
            for method in methods:
                trajs = grouped_distances[method].get(key, [])
                if not trajs:
                    continue
                max_len = max(len(d) for d in trajs)
                padded = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in trajs]
                arr = np.vstack(padded)
                mean_curve = np.nanmean(arr, axis=0)
                ax.plot(mean_curve, label=method, color=colors[method], lw=2)
                plotted = True
            if not plotted:
                ax.axis("off")
                continue

            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_title(f"r={r}, θ={angle}°", fontsize=10)

            # 保留 y 轴刻度
            if j == 0:
                ax.set_ylabel("Distance")
            if i == len(r_values) - 1:
                ax.set_xlabel("Time Steps")
            ax.yaxis.set_visible(True)  # 确保刻度可见

    # 统一图例放右上角
    fig.legend(methods, loc="upper right", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(colored(f"✅ Saved figure: {save_path}", "yellow"))
    plt.close(fig)

# ============================================================
# 4. 绘制成功率对比图
# ============================================================
def plot_success_rate(results_all, methods, save_path=None, min_steps=200):
    """
    绘制成功率对比图
    """
    success_rates = []
    for method in methods:
        traj_list = results_all[method]
        grouped = defaultdict(lambda: {"total":0, "success":0})
        for traj_data in traj_list:
            r = traj_data["r"]
            angle = traj_data["angle"]
            grouped[(r, angle)]["total"] += 1
            if len(traj_data["trajectory"]) >= min_steps:
                grouped[(r, angle)]["success"] += 1
        for key, stats in grouped.items():
            r, angle = key
            success_rates.append({
                "method": method,
                "r": r,
                "angle": angle,
                "success_rate": stats["success"] / stats["total"] * 100
            })

    df_success = pd.DataFrame(success_rates)
    fig, ax = plt.subplots(figsize=(10,6))
    for method in methods:
        df_m = df_success[df_success["method"]==method]
        ax.plot(range(len(df_m)), df_m["success_rate"], label=method, lw=2)
    ax.set_xticks(range(len(df_m)))
    ax.set_xticklabels([f"r={r},θ={angle}" for r,angle in zip(df_m["r"], df_m["angle"])], rotation=45)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("(r, θ) Combinations")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(colored(f"✅ Saved success-rate figure: {save_path}", "yellow"))
    # plt.show()
    plt.close(fig)

# def plot_hover_stability(df_metrics, methods, colors, save_dir="./hover_test_results"):
#     """
#     绘制各方法在不同 (r, θ) 任务上的距离、速度、动作幅度和平均角速度指标对比
#     每个小柱子表示：最小值、均值、最大值范围（小矩形），中间用细线连接各方法
#     第一行单子图居中，左右隐藏，第一行单子图y轴刻度正常，图例颜色对应方法
#     """
#     import os
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import matplotlib.patches as mpatches

#     os.makedirs(save_dir, exist_ok=True)

#     metrics_to_plot = ["mean_distance", "final_distance", "max_distance", 
#                        "mean_speed", "mean_action_norm", "mean_omega"]
#     r_values_all = sorted(df_metrics["r"].unique())
#     angle_values_all = sorted(df_metrics["angle"].unique())

#     for metric in metrics_to_plot:
#         n_rows = len(r_values_all)
#         n_cols = max(len(angle_values_all), 1)

#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=False)
#         if n_rows == 1 and n_cols == 1:
#             axes = np.array([[axes]])
#         elif n_rows == 1:
#             axes = axes[np.newaxis, :]
#         elif n_cols == 1:
#             axes = axes[:, np.newaxis]

#         for i, r in enumerate(r_values_all):
#             # 第一行 r=0 单子图居中
#             if r == 0:
#                 angle_values = [0]
#                 mid_idx = n_cols // 2
#             else:
#                 angle_values = angle_values_all

#             for j in range(n_cols):
#                 if r == 0:
#                     if j != mid_idx:
#                         axes[i, j].axis("off")
#                         continue
#                     ax = axes[i, j]
#                     angle = 0
#                 else:
#                     ax = axes[i, j] if n_rows > 1 else axes[j]
#                     angle = angle_values[j]

#                 y_mean, y_min, y_max = [], [], []
#                 for method in methods:
#                     df_sel = df_metrics[(df_metrics["method"]==method) & 
#                                         (df_metrics["r"]==r) & 
#                                         (df_metrics["angle"]==angle)]
#                     if len(df_sel) == 0:
#                         y_mean.append(np.nan)
#                         y_min.append(np.nan)
#                         y_max.append(np.nan)
#                     else:
#                         vals = df_sel[metric].values
#                         y_mean.append(np.mean(vals))
#                         y_min.append(np.min(vals))
#                         y_max.append(np.max(vals))

#                 # 绘制小矩形表示 min-max 范围
#                 width = 0.3  # 小矩形宽度
#                 for k, method in enumerate(methods):
#                     if not np.isnan(y_mean[k]):
#                         ax.add_patch(
#                             plt.Rectangle((k - width/2, y_min[k]), width, y_max[k]-y_min[k],
#                                           color=colors[method], alpha=0.3, zorder=2)
#                         )

#                 # 中间细线连接均值
#                 ax.plot(range(len(methods)), y_mean, color='k', lw=1.5, zorder=3, marker='o')

#                 ax.set_title(f"r={r}, θ={angle}°")
#                 ax.grid(True, linestyle='--', alpha=0.3)
#                 if j == 0 or (r==0 and j==mid_idx):
#                     ax.set_ylabel(metric)

#         # 图例
#         handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
#         fig.legend(handles=handles, labels=methods, loc="upper right", fontsize=10)

#         plt.tight_layout()
#         save_path = os.path.join(save_dir, f"hover_{metric}_comparison.png")
#         plt.savefig(save_path, dpi=300)
#         plt.close(fig)
#         print(f"✅ Saved {metric} comparison figure: {save_path}")


def plot_hover_stability(df_metrics, methods, colors, save_dir="./hover_test_results"):
    """
    绘制各方法在不同 (r, θ) 任务上的指标对比：
    - distance, speed, action norm, omega, distance variance
    - 每个小柱子表示最小值-最大值范围（小矩形），中间细线连接均值
    并保存整体汇总 CSV，统计每个方法的平均 final_distance / mean_omega / distance_var
    以及最大 final_distance / mean_omega / distance_var
    """
    os.makedirs(save_dir, exist_ok=True)

    metrics_to_plot = [
        "mean_distance", "final_distance", "max_distance",
        "mean_speed", "mean_action_norm", "mean_omega", "distance_var"
    ]
    r_values_all = sorted(df_metrics["r"].unique())
    angle_values_all = sorted(df_metrics["angle"].unique())

    # -------------------------------
    # 绘图
    # -------------------------------
    for metric in metrics_to_plot:
        n_rows = len(r_values_all)
        n_cols = max(len(angle_values_all), 1)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=False)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for i, r in enumerate(r_values_all):
            # r=0 第一行居中
            if r == 0:
                angle_values = [0]
                mid_idx = n_cols // 2
            else:
                angle_values = angle_values_all

            for j in range(n_cols):
                if r == 0 and j != mid_idx:
                    axes[i, j].axis("off")
                    continue
                ax = axes[i, j] if not (r==0 and j==mid_idx) else axes[i, j]
                angle = 0 if r==0 else angle_values[j]

                y_mean, y_min, y_max = [], [], []
                for method in methods:
                    df_sel = df_metrics[
                        (df_metrics["method"] == method) & 
                        (df_metrics["r"] == r) & 
                        (df_metrics["angle"] == angle)
                    ]
                    if len(df_sel) == 0:
                        y_mean.append(np.nan)
                        y_min.append(np.nan)
                        y_max.append(np.nan)
                    else:
                        vals = df_sel[metric].values
                        y_mean.append(np.mean(vals))
                        y_min.append(np.min(vals))
                        y_max.append(np.max(vals))

                # 绘制小矩形表示 min-max
                width = 0.3
                for k, method in enumerate(methods):
                    if not np.isnan(y_mean[k]):
                        ax.add_patch(
                            plt.Rectangle(
                                (k - width/2, y_min[k]),
                                width, y_max[k]-y_min[k],
                                color=colors[method], alpha=0.3, zorder=2
                            )
                        )

                # 中间细线连接均值
                ax.plot(range(len(methods)), y_mean, color='k', lw=1.5, zorder=3, marker='o')

                ax.set_title(f"r={r}, θ={angle}°")
                ax.grid(True, linestyle='--', alpha=0.3)
                if j == 0 or (r==0 and j==mid_idx):
                    ax.set_ylabel(metric)

        # 图例
        handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
        fig.legend(handles=handles, labels=methods, loc="upper right", fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"hover_{metric}_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved {metric} comparison figure: {save_path}")

    # -------------------------------
    # 汇总 CSV（整体统计，不区分 r / θ）
    # -------------------------------
    summary_rows = []
    for method in methods:
        df_sel = df_metrics[df_metrics["method"] == method]
        summary_rows.append({
            "method": method,
            "mean_final_distance": df_sel["final_distance"].mean(),
            "max_final_distance": df_sel["final_distance"].max(),
            "mean_omega": df_sel["mean_omega"].mean(),
            "max_omega": df_sel["mean_omega"].max(),
            "mean_distance_var": df_sel["distance_var"].mean(),
            "max_distance_var": df_sel["distance_var"].max()
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(save_dir, "hover_summary_overall.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"📊 Saved overall summary CSV: {csv_path}")






# ============================================================
# 5. 主流程
# ============================================================
if __name__ == "__main__":
    # save_paths = {
    #     "Controller": "./hover_test_results/hover_results.pkl",
    #     "RL": "./hover_test_results/hover_results_rl.pkl",
    #     "HRL": "./hover_test_results/hover_results_planner.pkl"
    # }

    # # 载入轨迹数据
    # results_all = load_trajectories(save_paths)
    # methods = list(save_paths.keys())
    # colors = {"Controller":"red", "RL":"blue", "HRL":"green"}

    # # 统计指标
    # # df_metrics, grouped_distances = compute_metrics(results_all, min_steps=200)
    # df_metrics, grouped_distances = compute_and_save_metrics(results_all, min_steps=200)

    # 保存表格
    # csv_save_path = "./hover_test_results/hover_metrics_comparison.csv"
    # df_metrics.to_csv(csv_save_path, index=False)
    # print(colored(f"✅ Saved metrics table: {csv_save_path}", "yellow"))

    # # 绘制均值距离对比图
    # fig_mean_path = "./hover_test_results/hover_distance_comparison.png"
    # plot_mean_distance(grouped_distances, methods, colors, save_path=fig_mean_path)

    # # 绘制成功率对比图
    # fig_success_path = "./hover_test_results/hover_success_rate.png"
    # plot_success_rate(results_all, methods, save_path=fig_success_path, min_steps=200)

    # methods = ["Controller", "RL", "HRL"]
    # colors = {"Controller":"red", "RL":"blue", "HRL":"green"}
    # csv_save_path = "./hover_test_results/hover_success_rate.csv"
    # df_metrics = pd.read_csv(csv_save_path)
    # plot_hover_stability(df_metrics, methods, colors)

    # 调用示例
    csv_save_path = "./hover_test_results/hover_success_rate.csv"
    plot_hover_success_rate_by_r(csv_save_path)
