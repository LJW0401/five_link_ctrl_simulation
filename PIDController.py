"""
PID 倒立摆控制器（IK + 关节位置 PID + 级联轮子控制）
"""

import math
from Controller import PID
from VMC import leg_VMC


class JOINT_PID:
    KP = 80.0
    KI = 1.0
    KD = 240.0
    INTEGRAL_LIMIT = 5.0
    OUTPUT_LIMIT = 20.0

class PITCH_PID:
    """内环：pitch → 轮子力矩"""
    KP = 25.0
    KI = 0.2
    KD = 100.0
    INTEGRAL_LIMIT = 1.0
    OUTPUT_LIMIT = 4.0

class POS_PID:
    """外环：位移/速度 → pitch 目标"""
    KP_X = 1.0
    KI_X = 0.0
    KP_V = 6.0
    INTEGRAL_LIMIT = 0.1
    OUTPUT_LIMIT = 0.4


class PIDBalanceController:
    """动态 IK + 关节 PID + 倒立摆轮子控制"""

    def __init__(self):
        self.L0_target = 0.25
        self.theta_target = 0.0

        self.theta_amplitude = 0.0
        self.theta_frequency = 1.0
        self.tick = 0

        self.vmc = leg_VMC()

        self.pid_joint = [
            PID(p=JOINT_PID.KP, i=JOINT_PID.KI, d=JOINT_PID.KD,
                integral_limit=JOINT_PID.INTEGRAL_LIMIT, output_limit=JOINT_PID.OUTPUT_LIMIT)
            for _ in range(4)
        ]

        self.pid_pitch = PID(p=PITCH_PID.KP, i=PITCH_PID.KI, d=PITCH_PID.KD,
                             integral_limit=PITCH_PID.INTEGRAL_LIMIT, output_limit=PITCH_PID.OUTPUT_LIMIT)

        self.pid_x = PID(p=POS_PID.KP_X, i=POS_PID.KI_X, d=0.0,
                         integral_limit=POS_PID.INTEGRAL_LIMIT, output_limit=POS_PID.OUTPUT_LIMIT)

        self.joint_targets = [0.0, 0.0, 0.0, 0.0]
        self.pitch_ref = 0.0

    def compute(self, joint_pos, pitch, body_x, body_vx=0.0, gyro_y=0.0):
        self.tick += 1
        t = self.tick * 0.004
        self.theta_target = self.theta_amplitude * math.sin(2 * math.pi * self.theta_frequency * t)

        phi0_r = math.pi / 2.0
        phi0_l = math.pi / 2.0

        ik_r = self.vmc.vmc_inverse_kinematics(self.L0_target, phi0_r)
        ik_l = self.vmc.vmc_inverse_kinematics(self.L0_target, phi0_l)

        if ik_r is not None:
            phi1_r, phi4_r = ik_r
            self.joint_targets[0] = phi1_r - math.pi
            self.joint_targets[1] = phi4_r
        else:
            print(f"[警告] 右腿 IK 无解: L0={self.L0_target:.3f}, phi0={phi0_r:.3f}")

        if ik_l is not None:
            phi1_l, phi4_l = ik_l
            self.joint_targets[2] = phi4_l
            self.joint_targets[3] = phi1_l - math.pi
        else:
            print(f"[警告] 左腿 IK 无解: L0={self.L0_target:.3f}, phi0={phi0_l:.3f}")

        joint_torque = [0.0] * 4
        for i in range(4):
            joint_torque[i] = self.pid_joint[i].calc(joint_pos[i], self.joint_targets[i])

        t_x = self.pid_x.calc(body_x, 0.0)
        t_v = POS_PID.KP_V * (0.0 - body_vx)
        self.pitch_ref = max(-POS_PID.OUTPUT_LIMIT, min(POS_PID.OUTPUT_LIMIT, t_x + t_v))

        t_pitch = -self.pid_pitch.calc(pitch, self.pitch_ref)
        wheel_torque = max(-4.0, min(4.0, t_pitch))

        return joint_torque, wheel_torque
