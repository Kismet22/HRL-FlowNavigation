import numpy as np
from numpy import *
from scipy.linalg import solve_continuous_are


class heading_controller(object):
    def __init__(self, sample_time, factor=None, speed_adaption=.3, max_controll=10):
        self.sample_time = sample_time  # 采样时间
        # TODO:暂时不需要step_adaption
        self.speed_adaption = speed_adaption
        self.factor = 1  # 使用一个简化的系数
        self.max_controll = max_controll  # 最大舵角

        # controller params:
        self.KP = 10
        self.KI = 3
        self.KD = 15
        # self.KP = 5
        # self.KI = 1
        # self.KD = 30


        # initial values:
        self.summed_error = 0  # 误差的积分

    # 航向参数计算
    def calculate_controller_params(self, yaw_time_constant=None, store=True, Q=None, r=None):
        # TODO:旋转响应先设置为1试试
        if yaw_time_constant is None:
            try:
                yaw_time_constant = YAW_TIMECONSTANT  # 时间常数
            except:
                raise Exception("Yaw time constant is required.")

        # 系统线性化建模
        A = array([[0, 1, 0], [0, -1. / yaw_time_constant, 0], [-1, 0, 0]])  # 状态矩阵
        B = array([0, 1, 1])  # 输入，舵角

        if Q is None:
            Q = diag([1E-1, 1, 0.3])  # 权重矩阵
            r = ones((1, 1)) * 30  # 期望代价

        # 通过Riccati方程计算反馈增益
        P = solve_continuous_are(A, B[:, None], Q, r)
        K = sum(B[None, :] * P, axis=1) / r[0, 0]

        if store:
            # 设置PID参数
            self.KP = K[0]
            self.KD = K[1]
            self.KI = -K[2]

        return list(K)

    def controll(self, desired_heading, heading, yaw_rate, speed=1, drift_angle=0):
        # 比例误差：期望航向 - 当前航向
        heading_error = desired_heading - heading

        # 确保误差在 [-pi, pi] 范围内
        while heading_error > pi:
            heading_error -= 2 * pi
        while heading_error < -pi:
            heading_error += 2 * pi

        # 积分误差（含偏移量）
        self.summed_error += self.sample_time * (heading_error - drift_angle)

        # 物理建模系数
        # if speed < self.speed_adaption:
        #     speed = self.speed_adaption
        # factor2 = -1. / self.factor / speed ** 2
        factor2 = 1

        # 计算舵角加速度（控制量为舵角的二阶导数）
        # 通过PID控制公式计算舵角加速度
        rudder_angle_acceleration = factor2 * (self.KP * heading_error + self.KI * self.summed_error - self.KD * yaw_rate)

        # 控制量超出限制范围
        if abs(rudder_angle_acceleration) > self.max_controll:
            rudder_angle_acceleration = sign(rudder_angle_acceleration) * self.max_controll
            self.summed_error = (rudder_angle_acceleration / factor2 - (self.KP * heading_error - self.KD * yaw_rate)) / self.KI

        return rudder_angle_acceleration

class PointController(object):
    def __init__(self, sample_time, max_acc=15.0, max_yaw_acc=5.0, 
                 limit_mode="component", fixed_heading=None):
        """
        目标点控制器：输出 x,y 加速度 + 航向角加速度
        :param sample_time: 控制器采样周期
        :param max_acc: 线加速度限幅 (模长 or 分量)
        :param max_yaw_acc: 角加速度限幅
        :param limit_mode: "component" -> 分量限幅, "norm" -> 模长限幅
        :param fixed_heading: 固定目标角度 (rad)，如果是 None 则用目标点方向计算
        """
        self.sample_time = sample_time
        self.max_acc = max_acc
        self.max_yaw_acc = max_yaw_acc
        self.limit_mode = limit_mode
        self.fixed_heading = fixed_heading  # 固定目标角度

        # PID 参数
        self.KP_pos = 2.0
        self.KI_pos = 0.05
        self.KD_pos = 0.8

        self.KP_yaw = 15.0
        self.KI_yaw = 0.1
        self.KD_yaw = 2.0

        # 积分项
        self.sum_error_pos = np.zeros(2)  # x,y 位置积分误差
        self.sum_error_yaw = 0.0

    def control(self, target_pos, agent_pos, agent_heading, agent_vel, yaw_rate):
        # --- 1. 位置控制 ---
        pos_error = target_pos - agent_pos  # [dx, dy]
        self.sum_error_pos += pos_error * self.sample_time
        self.sum_error_pos = np.clip(self.sum_error_pos, -5.0, 5.0)

        acc_cmd = (
            self.KP_pos * pos_error
            + self.KI_pos * self.sum_error_pos
            - self.KD_pos * agent_vel
        )

        # --- 限幅方式选择 ---
        if self.limit_mode == "component":
            acc_cmd = np.clip(acc_cmd, -self.max_acc, self.max_acc)
        elif self.limit_mode == "norm":
            acc_norm = np.linalg.norm(acc_cmd)
            if acc_norm > self.max_acc:
                acc_cmd = acc_cmd / acc_norm * self.max_acc

        # --- 2. 航向控制 ---
        if self.fixed_heading is not None:
            desired_heading = self.fixed_heading   # 使用固定参考角度
        else:
            desired_heading = np.arctan2(pos_error[1], pos_error[0])  # 默认动态角度

        heading_error = desired_heading - agent_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        self.sum_error_yaw += heading_error * self.sample_time
        self.sum_error_yaw = np.clip(self.sum_error_yaw, -2.0, 2.0)

        yaw_acc_cmd = (
            self.KP_yaw * heading_error
            + self.KI_yaw * self.sum_error_yaw
            - self.KD_yaw * yaw_rate
        )

        yaw_acc_cmd = np.clip(yaw_acc_cmd, -self.max_yaw_acc, self.max_yaw_acc)

        return np.array([acc_cmd[0], acc_cmd[1], yaw_acc_cmd])


