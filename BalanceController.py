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
    KP_X = 1.0      # 位移 → pitch 偏移
    KI_X = 0.0
    KP_V = 6.0      # 速度 → pitch 偏移
    INTEGRAL_LIMIT = 0.1  # pitch 偏移限幅 (rad)
    OUTPUT_LIMIT = 0.4   # 最大 pitch 目标 (rad)


class BalanceController:
    """动态 IK + 关节 PID + 倒立摆轮子控制"""

    def __init__(self):
        # 目标值
        self.L0_target = 0.25   # 目标腿长 (m)
        self.theta_target = 0.0 # 目标摆杆夹角 (rad)

        # theta 正弦波参数（设 amplitude=0 关闭）
        self.theta_amplitude = 0.0
        self.theta_frequency = 1.0
        self.tick = 0

        # VMC 实例（用于 IK）
        self.vmc = leg_VMC()

        # 关节位置 PID（4个关节）
        self.pid_joint = [
            PID(p=JOINT_PID.KP, i=JOINT_PID.KI, d=JOINT_PID.KD,
                integral_limit=JOINT_PID.INTEGRAL_LIMIT, output_limit=JOINT_PID.OUTPUT_LIMIT),
            PID(p=JOINT_PID.KP, i=JOINT_PID.KI, d=JOINT_PID.KD,
                integral_limit=JOINT_PID.INTEGRAL_LIMIT, output_limit=JOINT_PID.OUTPUT_LIMIT),
            PID(p=JOINT_PID.KP, i=JOINT_PID.KI, d=JOINT_PID.KD,
                integral_limit=JOINT_PID.INTEGRAL_LIMIT, output_limit=JOINT_PID.OUTPUT_LIMIT),
            PID(p=JOINT_PID.KP, i=JOINT_PID.KI, d=JOINT_PID.KD,
                integral_limit=JOINT_PID.INTEGRAL_LIMIT, output_limit=JOINT_PID.OUTPUT_LIMIT),
        ]

        # 内环：pitch PID → 轮子力矩
        self.pid_pitch = PID(p=PITCH_PID.KP, i=PITCH_PID.KI, d=PITCH_PID.KD,
                             integral_limit=PITCH_PID.INTEGRAL_LIMIT, output_limit=PITCH_PID.OUTPUT_LIMIT)

        # 外环：位移 PID → pitch 目标
        self.pid_x = PID(p=POS_PID.KP_X, i=POS_PID.KI_X, d=0.0,
                         integral_limit=POS_PID.INTEGRAL_LIMIT, output_limit=POS_PID.OUTPUT_LIMIT)

        # 当前关节目标（供外部监控）
        self.joint_targets = [0.0, 0.0, 0.0, 0.0]
        self.pitch_ref = 0.0  # 供外部监控

    def compute(self, joint_pos, pitch, body_x, body_vx=0.0):
        """
        返回:
            joint_torque: [右前, 右后, 左前, 左后]
            wheel_torque: 左右相同
        """
        # --- theta 正弦波 ---
        self.tick += 1
        t = self.tick * 0.004
        self.theta_target = self.theta_amplitude * math.sin(2 * math.pi * self.theta_frequency * t)

        # --- phi0 固定相对机体（不随 pitch 变化，让轮子控制 pitch）---
        phi0_r = math.pi / 2.0
        phi0_l = math.pi / 2.0

        # --- IK 解算 ---
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

        # --- 关节 PID ---
        joint_torque = [0.0] * 4
        for i in range(4):
            joint_torque[i] = self.pid_joint[i].calc(joint_pos[i], self.joint_targets[i])

        # --- 倒立摆轮子控制 ---
        # 外环：位移/速度 → pitch 目标
        t_x = self.pid_x.calc(body_x, 0.0)
        t_v = POS_PID.KP_V * (0.0 - body_vx)
        self.pitch_ref = max(-POS_PID.OUTPUT_LIMIT, min(POS_PID.OUTPUT_LIMIT, t_x + t_v))

        # 内环：pitch → 轮子力矩
        t_pitch = -self.pid_pitch.calc(pitch, self.pitch_ref)
        wheel_torque = max(-4.0, min(4.0, t_pitch))

        return joint_torque, wheel_torque
