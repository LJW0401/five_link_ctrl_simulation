import math
from Controller import PID
from VMC import leg_VMC


class BalanceController:
    """基于动态 IK + 关节位置 PID 的控制器"""

    def __init__(self):
        # 目标值
        self.L0_target = 0.15   # 目标腿长 (m)
        self.theta_target = 0.0 # 目标摆杆夹角 (rad)，实时更新

        # theta 正弦波参数
        self.theta_amplitude = 0.3  # 幅值 (rad)
        self.theta_frequency = 1.0  # 频率 (Hz)
        self.tick = 0

        # VMC 实例（用于 IK）
        self.vmc = leg_VMC()

        # 关节位置 PID（4个关节：右前, 右后, 左前, 左后）
        self.pid_joint = [
            PID(p=20.0, i=0.3, d=30.0, integral_limit=20.0, output_limit=20.0),
            PID(p=20.0, i=0.3, d=30.0, integral_limit=20.0, output_limit=20.0),
            PID(p=20.0, i=0.3, d=30.0, integral_limit=20.0, output_limit=20.0),
            PID(p=20.0, i=0.3, d=30.0, integral_limit=20.0, output_limit=20.0),
        ]

        # 当前关节目标（供外部监控）
        self.joint_targets = [0.0, 0.0, 0.0, 0.0]

    def compute(self, joint_pos, pitch, gyro_y, body_x, body_vx):
        """
        每个控制周期：
        1. 由 (L0_target, theta_target, pitch) 计算 phi0
        2. IK 求目标关节角
        3. 关节 PID 跟踪

        返回:
            joint_torque: [右前, 右后, 左前, 左后]
            wheel_torque: 0（屏蔽）
        """
        # --- 更新 theta 正弦波目标 ---
        self.tick += 1
        t = self.tick * 0.004  # 控制周期 4ms
        self.theta_target = self.theta_amplitude * math.sin(2 * math.pi * self.theta_frequency * t)

        # --- 由 theta 和 pitch 计算目标 phi0 ---
        # 右腿: theta = pi/2 + pitch - phi0  → phi0 = pi/2 + pitch - theta
        phi0_r = math.pi / 2.0 + pitch - self.theta_target
        # 左腿(镜像): theta = pi/2 - pitch - phi0  → phi0 = pi/2 - pitch - theta
        phi0_l = math.pi / 2.0 - pitch - self.theta_target

        # --- IK 解算 ---
        ik_r = self.vmc.vmc_inverse_kinematics(self.L0_target, phi0_r)
        ik_l = self.vmc.vmc_inverse_kinematics(self.L0_target, phi0_l)

        if ik_r is not None:
            phi1_r, phi4_r = ik_r
            self.joint_targets[0] = phi1_r - math.pi  # 右前
            self.joint_targets[1] = phi4_r             # 右后

        if ik_l is not None:
            phi1_l, phi4_l = ik_l
            self.joint_targets[2] = phi4_l             # 左前（对应左腿 phi4）
            self.joint_targets[3] = phi1_l - math.pi   # 左后（对应左腿 phi1）

        # --- 关节 PID ---
        joint_torque = [0.0] * 4
        for i in range(4):
            joint_torque[i] = self.pid_joint[i].calc(joint_pos[i], self.joint_targets[i])

        return joint_torque, 0.0
