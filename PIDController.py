"""
PID 倒立摆控制器（IK + 关节位置 PID + 级联轮子控制）
"""

import math
from Controller import PID
from VMC import leg_VMC
from StateEstimator import StateEstimator


class JOINT_PID:
    KP = 100.0
    KI = 1.0
    KD = 300.0
    INTEGRAL_LIMIT = 5.0
    OUTPUT_LIMIT = 20.0

class PITCH_PID:
    """内环：pitch → 轮子力矩"""
    KP = 12.0
    KI = 0.2
    KD = 100.0
    INTEGRAL_LIMIT = 2.0
    OUTPUT_LIMIT = 4.0

class POS_PID:
    """外环：位移 → pitch 目标"""
    KP_X = 0.05
    KI_X = 0.0001
    KP_V = 0.1
    INTEGRAL_LIMIT = 0.2
    OUTPUT_LIMIT = 0.3


class PIDBalanceController:
    """动态 IK + 关节 PID + 倒立摆轮子控制"""

    def __init__(self, leg_params=None):
        self.L0_target = 0.25
        self.theta_target = 0.0

        self.theta_amplitude = 0.0
        self.theta_frequency = 1.0
        self.tick = 0

        self.leg_params = leg_params
        self.state = StateEstimator(leg_params)
        self.vmc = leg_VMC(leg_params)

        self.pid_joint = [
            PID(p=JOINT_PID.KP, i=JOINT_PID.KI, d=JOINT_PID.KD,
                integral_limit=JOINT_PID.INTEGRAL_LIMIT, output_limit=JOINT_PID.OUTPUT_LIMIT)
            for _ in range(4)
        ]

        self.pid_pitch = PID(p=PITCH_PID.KP, i=PITCH_PID.KI, d=PITCH_PID.KD,
                             integral_limit=PITCH_PID.INTEGRAL_LIMIT, output_limit=PITCH_PID.OUTPUT_LIMIT)

        self.pid_x = PID(p=POS_PID.KP_X, i=POS_PID.KI_X, d=POS_PID.KP_V,
                         integral_limit=POS_PID.INTEGRAL_LIMIT, output_limit=POS_PID.OUTPUT_LIMIT)

        self.joint_targets = [0.0, 0.0, 0.0, 0.0]
        self.pitch_ref = 0.0

    def compute(self, imu, motors):
        """
        参数:
            imu:    IMUData — 机体 IMU 数据
            motors: list[MotorData] — 6 个电机数据
                    顺序: [右前关节, 右后关节, 左前关节, 左后关节, 右轮, 左轮]
        返回:
            joint_torque: [右前, 右后, 左前, 左后]
            wheel_torque: [右轮, 左轮]
        """
        self.state.update(imu, motors)

        joint_pos = [motors[j].pos for j in range(4)]
        # StateEstimator 约定 phi = -pitch，保持原控制律极性
        pitch = -self.state.body.phi
        body_x = self.state.body.x
        body_vx = self.state.body.x_dot

        self.tick += 1
        t = self.tick * 0.004
        self.theta_target = self.theta_amplitude * math.sin(2 * math.pi * self.theta_frequency * t)

        phi0_r = math.pi / 2.0
        phi0_l = math.pi / 2.0

        ik_r = self.vmc.calc_inverse_kinematics(self.L0_target, phi0_r)
        ik_l = self.vmc.calc_inverse_kinematics(self.L0_target, phi0_l)

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
        self.pitch_ref = max(-POS_PID.OUTPUT_LIMIT, min(POS_PID.OUTPUT_LIMIT, t_x))

        print(f"pitch={pitch:.3f} rad, pitch_ref={self.pitch_ref:.3f} rad | body_x={body_x:.3f} m, body_vx={body_vx:.3f} m/s")

        t_pitch = -self.pid_pitch.calc(pitch, self.pitch_ref)
        t_wheel = max(-4.0, min(4.0, t_pitch))
        wheel_torque = [t_wheel, t_wheel]

        return joint_torque, wheel_torque
