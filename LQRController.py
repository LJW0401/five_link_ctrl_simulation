"""
LQR 状态反馈控制器

启动时从 lqr_config.json 加载二次多项式系数，
运行时根据当前腿长 L0 计算 K = a2*L0² + a1*L0 + a0。

状态向量 x[6]: [theta, d_theta, x, d_x, phi, d_phi]
控制输出: T (轮子力矩), Tp (髋关节力矩)
"""

import math
import json
import os
from StateEstimator import StateEstimator, IMUData, MotorData

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "lqr_config.json")


class KTable:
    """K 矩阵查找表，线性插值"""

    def __init__(self, L0_values, K_table):
        self.L0_values = L0_values
        self.K_table = K_table  # list of list: [n][2][6]
        self.n = len(L0_values)
        self.L0_min = L0_values[0]
        self.L0_max = L0_values[-1]

    def get_k(self, L0):
        """根据 L0 线性插值获取 K[2][6]"""
        # 钳位
        if L0 <= self.L0_min:
            return [row[:] for row in self.K_table[0]]
        if L0 >= self.L0_max:
            return [row[:] for row in self.K_table[-1]]

        # 找区间
        step = (self.L0_max - self.L0_min) / (self.n - 1)
        idx = int((L0 - self.L0_min) / step)
        idx = min(idx, self.n - 2)

        # 插值
        t = (L0 - self.L0_values[idx]) / (self.L0_values[idx + 1] - self.L0_values[idx])
        k = [[0.0] * 6 for _ in range(2)]
        for i in range(2):
            for j in range(6):
                k[i][j] = (1 - t) * self.K_table[idx][i][j] + t * self.K_table[idx + 1][i][j]
        return k


def calc_lqr(k, x):
    """矩阵相乘: T_Tp = K * x"""
    T  = sum(k[0][j] * x[j] for j in range(6))
    Tp = sum(k[1][j] * x[j] for j in range(6))
    return T, Tp


class LQRBalanceController:
    """
    LQR 平衡控制器

    控制流程:
      1. StateEstimator 获取状态
      2. 查找表线性插值获取 K(L0)
      3. LQR 状态反馈 → T, Tp
      4. PID 控制腿长 → F0
      5. VMC 雅可比 (F0, Tp) → 关节力矩
    """

    def __init__(self, config_path=CONFIG_FILE):
        from Controller import PID

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"未找到 {config_path}，请先运行 python calc_lqr_k.py 生成")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.k_table = KTable(config["L0_values"], config["K_table"])
        self.leg_params = config.get("leg_params", None)
        L0_range = config["L0_range"]
        n = len(config["L0_values"])
        print(f"[LQR] 加载K矩阵查找表: {n}点, L0 ∈ [{L0_range['min']:.2f}, {L0_range['max']:.2f}]m")

        # 目标值
        self.L0_target = 0.18
        self.x_target = 0.0
        self.v_target = 0.0

        # 状态估计器（传入五连杆参数）
        self.state = StateEstimator(self.leg_params)

        # 腿长 PID
        self.pid_L0_r = PID(p=2000.0, i=10.0, d=9000.0, integral_limit=50.0, output_limit=300.0)
        self.pid_L0_l = PID(p=2000.0, i=10.0, d=9000.0, integral_limit=50.0, output_limit=300.0)

        # 监控
        self.T = 0.0
        self.Tp_r = 0.0
        self.Tp_l = 0.0

        # 计数器
        self.t_count = 0

    def compute(self, imu, motors):
        """
        参数:
            imu:    IMUData — 机体 IMU 数据
            motors: list[MotorData] — 6 个电机数据
                    顺序: [右前关节, 右后关节, 左前关节, 左后关节, 右轮, 左轮]
        返回:
            joint_torque: [右前, 右后, 左前, 左后]
            wheel_torque: 左右相同
        """
        self.t_count += 1

        self.state.update(imu, motors)

        phi = self.state.body.phi
        phi_dot = self.state.body.phi_dot
        body_x = self.state.body.x
        body_vx = self.state.body.x_dot

        # --- LQR ---
        wheel_torque_sum = 0.0
        for i in range(2):
            leg = self.state.leg[i]
            k = self.k_table.get_k(leg.L0)

            x = [
                leg.Theta,
                leg.dTheta,
                body_x - self.x_target,
                body_vx - self.v_target,
                phi,
                phi_dot,
            ]

            x = [
                leg.Theta - 0.3 * math.sin(self.t_count * 0.01),  # 以站立姿态为目标
                leg.dTheta,
                0,0,0,0]
            
            # x = [0,0,0,0,0,0]  # --- IGNORE ---
            

            T, Tp = calc_lqr(k, x)
            wheel_torque_sum += T

            if i == 0:
                self.Tp_r = Tp
            else:
                self.Tp_l = Tp

        self.T = wheel_torque_sum / 2.0
        wheel_torque = max(-4.0, min(4.0, self.T))
        wheel_torque = 0

        # --- 腿长 PID + 重力前馈 ---
        ff_r = StateEstimator.gravity_feedforward(self.state.leg[0].Theta)
        F0_r = self.pid_L0_r.calc(self.state.leg[0].L0, self.L0_target) + ff_r

        ff_l = StateEstimator.gravity_feedforward(self.state.leg[1].Theta)
        F0_l = self.pid_L0_l.calc(self.state.leg[1].L0, self.L0_target) + ff_l

        # --- VMC ---
        self.state.vmc_r.calc_torque(F0_r, self.Tp_r)

        self.state.vmc_l.calc_torque(F0_l, self.Tp_l)

        joint_torque = [
            -self.state.vmc_r.torque_set[1], # 右前
            -self.state.vmc_r.torque_set[0], # 右后
            self.state.vmc_l.torque_set[1], # 左前
            self.state.vmc_l.torque_set[0], # 左后
        ]

        return joint_torque, wheel_torque
