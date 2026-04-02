"""
LQR 状态反馈控制器
参考: Modular_Balancing_Robot_Controller/application/chassis/chassis_balance.c

状态向量 x[6]:
  x[0]: theta     — 摆杆与竖直方向夹角
  x[1]: theta_dot — 摆杆角速度
  x[2]: x         — 位移
  x[3]: x_dot     — 速度
  x[4]: phi       — 机体倾角 (= -pitch)
  x[5]: phi_dot   — 机体角速度

控制输出:
  T  — 轮子力矩
  Tp — 髋关节力矩（VMC 虚拟力矩）

K 矩阵 [2×6] 通过 L0 的三次多项式拟合获得（MATLAB lqr 离线计算）
"""

import math
from StateEstimator import StateEstimator, BODY_MASS, GRAVITY


def get_k(L0):
    """
    根据腿长 L0 计算 LQR 增益矩阵 K[2][6]
    每个元素是 L0 的三次多项式: k = a3*L0^3 + a2*L0^2 + a1*L0 + a0

    参数:
        L0: 当前腿长 (m)
    返回:
        k: 2x6 列表, k[0][*] 为轮子力矩增益, k[1][*] 为髋关节力矩增益
    """
    t1 = L0
    t2 = L0 * L0
    t3 = L0 * L0 * L0

    k = [[0.0] * 6 for _ in range(2)]

    # 轮子力矩 T 的增益
    k[0][0] = -157.0203 * t3 + 179.8485 * t2 - 85.4172 * t1 + 0.0537
    k[0][1] =   -1.0899 * t3 +   3.0072 * t2 -  6.6735 * t1 + 0.1905
    k[0][2] =  -27.7037 * t3 +  27.7682 * t2 -  9.7108 * t1 - 0.2497
    k[0][3] =  -32.3706 * t3 +  32.7508 * t2 - 12.0579 * t1 - 0.3533
    k[0][4] =  -38.5250 * t3 +  54.5544 * t2 - 30.4327 * t1 + 8.2300
    k[0][5] =   -5.2760 * t3 +   8.9537 * t2 -  5.7839 * t1 + 1.8671

    # 髋关节力矩 Tp 的增益
    k[1][0] =  152.7053 * t3 - 130.2649 * t2 + 28.7526 * t1 + 5.6772
    k[1][1] =   16.2673 * t3 -  16.3037 * t2 +  5.2455 * t1 + 0.1397
    k[1][2] =  -21.5602 * t3 +  27.7447 * t2 - 13.9335 * t1 + 3.1997
    k[1][3] =  -24.4104 * t3 +  31.6150 * t2 - 16.0527 * t1 + 3.7713
    k[1][4] =  224.1583 * t3 - 227.6099 * t2 + 81.5551 * t1 + 0.1116
    k[1][5] =   50.1510 * t3 -  51.5745 * t2 + 18.8992 * t1 - 0.3485

    return k


def calc_lqr(k, x):
    """
    矩阵相乘: T_Tp = K * x

    参数:
        k: 2x6 增益矩阵
        x: 6 维状态向量
    返回:
        (T, Tp): 轮子力矩, 髋关节力矩
    """
    T  = sum(k[0][j] * x[j] for j in range(6))
    Tp = sum(k[1][j] * x[j] for j in range(6))
    return T, Tp


class LQRBalanceController:
    """
    LQR 平衡控制器

    控制流程:
      1. StateEstimator 获取状态 (theta, dTheta, x, vx, phi, phi_dot)
      2. LQR 状态反馈 → 轮子力矩 T 和髋关节力矩 Tp
      3. 腿长 L0 由 PID 控制 → F0
      4. VMC 雅可比将 (F0, Tp) 映射为关节力矩
    """

    def __init__(self):
        from Controller import PID

        # 目标值
        self.L0_target = 0.25
        self.x_target = 0.0
        self.v_target = 0.0

        # 状态估计器
        self.state = StateEstimator()

        # 腿长 PID（LQR 不控制腿长，用 PID 补充）
        self.pid_L0_r = PID(p=1000.0, i=20.0, d=60.0, integral_limit=50.0, output_limit=300.0)
        self.pid_L0_l = PID(p=1000.0, i=20.0, d=60.0, integral_limit=50.0, output_limit=300.0)

        # 监控变量
        self.T = 0.0
        self.Tp_r = 0.0
        self.Tp_l = 0.0
        self.F0_r = 0.0
        self.F0_l = 0.0

    def compute(self, joint_pos, pitch, body_x, body_vx=0.0, gyro_y=0.0):
        """
        参数:
            joint_pos: [右前, 右后, 左前, 左后]
            pitch:     俯仰角 (rad)
            body_x:    位移 (m)
            body_vx:   速度 (m/s)
            gyro_y:    俯仰角速度 (rad/s)
        返回:
            joint_torque: [右前, 右后, 左前, 左后]
            wheel_torque: 左右相同
        """
        # 更新状态
        self.state.update(joint_pos, pitch, gyro_y, body_x, body_vx)

        phi = -pitch
        phi_dot = -gyro_y

        # --- 对每条腿计算 LQR ---
        wheel_torque_sum = 0.0
        for i in range(2):
            leg = self.state.leg[i]
            L0 = leg.L0

            # 获取 K 矩阵
            k = get_k(L0)

            # 状态误差向量
            x = [
                leg.theta - 0.0,         # theta 目标 = 0
                leg.dTheta - 0.0,        # theta_dot 目标 = 0
                body_x - self.x_target,  # x 目标
                body_vx - self.v_target, # v 目标
                phi - 0.0,               # phi 目标 = 0
                phi_dot - 0.0,           # phi_dot 目标 = 0
            ]

            T, Tp = calc_lqr(k, x)
            wheel_torque_sum += T

            # 保存 Tp 到 VMC
            if i == 0:
                self.Tp_r = Tp
            else:
                self.Tp_l = Tp

        # 轮子力矩取两腿平均
        self.T = wheel_torque_sum / 2.0
        wheel_torque = max(-4.0, min(4.0, self.T))

        # --- 腿长 PID + 重力前馈 → F0 ---
        ff_r = StateEstimator.gravity_feedforward(self.state.leg[0].theta)
        self.F0_r = self.pid_L0_r.calc(self.state.leg[0].L0, self.L0_target) + ff_r

        ff_l = StateEstimator.gravity_feedforward(self.state.leg[1].theta)
        self.F0_l = self.pid_L0_l.calc(self.state.leg[1].L0, self.L0_target) + ff_l

        # --- VMC: (F0, Tp) → 关节力矩 ---
        self.state.vmc_r.F0 = self.F0_r
        self.state.vmc_r.Tp = self.Tp_r
        self.state.vmc_r.vmc_calc_torque()

        self.state.vmc_l.F0 = self.F0_l
        self.state.vmc_l.Tp = self.Tp_l
        self.state.vmc_l.vmc_calc_torque()

        joint_torque = [
            self.state.vmc_r.torque_set[1],  # 右前
            self.state.vmc_r.torque_set[0],  # 右后
            self.state.vmc_l.torque_set[0],  # 左前
            self.state.vmc_l.torque_set[1],  # 左后
        ]

        return joint_torque, wheel_torque
