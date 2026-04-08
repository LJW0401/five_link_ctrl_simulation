"""
LQR 状态反馈控制器

启动时从 lqr_config.json 加载 K 矩阵查找表，
运行时根据当前腿长 L0 线性插值获取 K。

状态向量 x[6]: [theta, d_theta, x, d_x, phi, d_phi]
控制输出: T (轮子力矩), Tp (髋关节力矩)
"""

import json
import os
import numpy as np
from StateEstimator import StateEstimator

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "lqr_config.json")


class KTable:
    """K 矩阵查找表，支持线性插值"""

    def __init__(self, config):
        self.L0_values = np.array(config["L0_values"])
        self.K_table = np.array(config["K_table"])  # shape: (n, 2, 6)
        self.L0_min = self.L0_values[0]
        self.L0_max = self.L0_values[-1]

    def get_k(self, L0):
        """根据 L0 插值获取 K[2][6]"""
        L0_clamped = np.clip(L0, self.L0_min, self.L0_max)
        # 找插值位置
        idx = np.searchsorted(self.L0_values, L0_clamped, side='right') - 1
        idx = np.clip(idx, 0, len(self.L0_values) - 2)

        # 线性插值
        L0_lo = self.L0_values[idx]
        L0_hi = self.L0_values[idx + 1]
        t = (L0_clamped - L0_lo) / (L0_hi - L0_lo)

        K = (1 - t) * self.K_table[idx] + t * self.K_table[idx + 1]
        return K.tolist()


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
      2. K 矩阵查找表插值
      3. LQR 状态反馈 → T, Tp
      4. PID 控制腿长 → F0
      5. VMC 雅可比 (F0, Tp) → 关节力矩
    """

    def __init__(self, config_path=CONFIG_FILE):
        from Controller import PID

        # 加载 K 矩阵配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"未找到 {config_path}，请先运行 python calc_lqr_k.py 生成")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.k_table = KTable(config)
        self.robot_params = config["robot_params"]

        print(f"[LQR] 加载 K 矩阵: {len(config['L0_values'])} 个采样点, "
              f"L0 ∈ [{config['L0_range']['min']:.2f}, {config['L0_range']['max']:.2f}]m")

        # 目标值
        self.L0_target = 0.25
        self.x_target = 0.0
        self.v_target = 0.0

        # 状态估计器
        self.state = StateEstimator()

        # 腿长 PID
        self.pid_L0_r = PID(p=1000.0, i=20.0, d=60.0, integral_limit=50.0, output_limit=300.0)
        self.pid_L0_l = PID(p=1000.0, i=20.0, d=60.0, integral_limit=50.0, output_limit=300.0)

        # 监控
        self.T = 0.0
        self.Tp_r = 0.0
        self.Tp_l = 0.0

    def compute(self, joint_pos, pitch, body_x, body_vx=0.0, gyro_y=0.0):
        """
        返回:
            joint_torque: [右前, 右后, 左前, 左后]
            wheel_torque: 左右相同
        """
        self.state.update(joint_pos, pitch, gyro_y, body_x, body_vx)

        phi = -pitch
        phi_dot = -gyro_y

        # --- LQR ---
        wheel_torque_sum = 0.0
        for i in range(2):
            leg = self.state.leg[i]

            k = self.k_table.get_k(leg.L0)

            x = [
                leg.theta - 0.0,
                leg.dTheta - 0.0,
                body_x - self.x_target,
                body_vx - self.v_target,
                phi - 0.0,
                phi_dot - 0.0,
            ]

            T, Tp = calc_lqr(k, x)
            wheel_torque_sum += T

            if i == 0:
                self.Tp_r = Tp
            else:
                self.Tp_l = Tp

        self.T = wheel_torque_sum / 2.0
        wheel_torque = max(-4.0, min(4.0, self.T))

        # --- 腿长 PID + 重力前馈 ---
        ff_r = StateEstimator.gravity_feedforward(self.state.leg[0].theta)
        F0_r = self.pid_L0_r.calc(self.state.leg[0].L0, self.L0_target) + ff_r

        ff_l = StateEstimator.gravity_feedforward(self.state.leg[1].theta)
        F0_l = self.pid_L0_l.calc(self.state.leg[1].L0, self.L0_target) + ff_l

        # --- VMC ---
        self.state.vmc_r.F0 = F0_r
        self.state.vmc_r.Tp = self.Tp_r
        self.state.vmc_r.vmc_calc_torque()

        self.state.vmc_l.F0 = F0_l
        self.state.vmc_l.Tp = self.Tp_l
        self.state.vmc_l.vmc_calc_torque()

        joint_torque = [
            self.state.vmc_r.torque_set[1],
            self.state.vmc_r.torque_set[0],
            self.state.vmc_l.torque_set[0],
            self.state.vmc_l.torque_set[1],
        ]

        return joint_torque, wheel_torque
