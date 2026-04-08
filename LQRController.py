"""
LQR 状态反馈控制器

启动时从 lqr_config.json 加载二次多项式系数，
运行时根据当前腿长 L0 计算 K = a2*L0² + a1*L0 + a0。

状态向量 x[6]: [theta, d_theta, x, d_x, phi, d_phi]
控制输出: T (轮子力矩), Tp (髋关节力矩)
"""

import json
import os
from StateEstimator import StateEstimator

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "lqr_config.json")


def load_poly_coeffs(config_path=CONFIG_FILE):
    """从 JSON 加载多项式系数"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["poly_coeffs"]


def get_k(L0, poly_coeffs):
    """
    用多项式计算 K[2][6]（Horner 法，支持任意阶数）
    """
    k = [[0.0] * 6 for _ in range(2)]
    for i in range(2):
        for j in range(6):
            c = poly_coeffs[i][j]
            val = 0.0
            for coeff in c:
                val = val * L0 + coeff
            k[i][j] = val
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
      2. 二次多项式计算 K(L0)
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

        self.poly_coeffs = config["poly_coeffs"]
        L0_range = config["L0_range"]
        print(f"[LQR] 加载二次多项式系数, L0 ∈ [{L0_range['min']:.2f}, {L0_range['max']:.2f}]m")

        # 目标值
        self.L0_target = 0.18
        self.x_target = 0.0
        self.v_target = 0.0

        # 状态估计器
        self.state = StateEstimator()

        # 腿长 PID
        self.pid_L0_r = PID(p=5000.0, i=20.0, d=3000.0, integral_limit=50.0, output_limit=300.0)
        self.pid_L0_l = PID(p=5000.0, i=20.0, d=3000.0, integral_limit=50.0, output_limit=300.0)

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
            k = get_k(leg.L0, self.poly_coeffs)

            x = [
                leg.theta,
                leg.dTheta,
                body_x - self.x_target,
                body_vx - self.v_target,
                phi,
                phi_dot,
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
