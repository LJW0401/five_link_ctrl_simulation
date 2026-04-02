import math
from Controller import PID
from VMC import leg_VMC


# 控制阶段
PHASE_FREEFALL = 0   # 自由落体，等待落地
PHASE_LEG_CTRL = 1   # 控制腿部到目标姿态
PHASE_BALANCE  = 2   # 全部控制，站立平衡


class BalanceController:
    """分阶段平衡控制器：落地 → 调腿 → 站立"""

    def __init__(self):
        # 目标虚拟腿参数
        self.L0_target = 0.2            # 目标腿长 (m)
        self.phi0_target = math.pi / 2  # 目标腿角 (rad)，竖直向下

        # VMC 实例（正/逆运动学）
        self.vmc_r = leg_VMC()
        self.vmc_l = leg_VMC()

        # 关节位置 PID（4个关节：右前, 右后, 左前, 左后）
        self.pid_joint = [
            PID(p=5, i=0.1, d=0, integral_limit=10.0, output_limit=20.0),
            PID(p=5, i=0.1, d=0, integral_limit=10.0, output_limit=20.0),
            PID(p=5, i=0.1, d=0, integral_limit=10.0, output_limit=20.0),
            PID(p=5, i=0.1, d=0, integral_limit=10.0, output_limit=20.0),
        ]

        # 平衡 PID（轮子力矩控制 pitch）
        self.pid_pitch = PID(p=100.0, i=5.0, d=0, integral_limit=10.0, output_limit=3.0)
        self.pid_gyro  = PID(p=15.0,  i=0.0, d=0.0,  integral_limit=10.0, output_limit=2.0)

        # 位移和速度 PID（轮子力矩保持原地）
        self.pid_x = PID(p=2.0,  i=0.05, d=0.0, integral_limit=10.0, output_limit=1.0)
        self.pid_v = PID(p=3.0,  i=0.0,  d=0.0, integral_limit=10.0, output_limit=1.5)

        # IK 目标关节角度
        self.joint_targets = [0.0, 0.0, 0.0, 0.0]

        # 阶段控制
        self.phase = PHASE_FREEFALL
        self.phase_tick = 0              # 当前阶段经过的控制周期数
        self.freefall_ticks = 200        # 落地等待时间（控制周期数，×4ms = 0.8s）
        self.leg_settle_threshold = 0.05 # 腿部关节误差阈值 (rad)

    def update_ik_targets(self):
        """用逆运动学计算目标关节角度"""
        result = self.vmc_r.vmc_inverse_kinematics(self.L0_target, self.phi0_target)
        if result is None:
            return False

        phi1_ik, phi4_ik = result

        # 转换到传感器关节角度坐标系
        self.joint_targets[0] = phi1_ik - math.pi   # 右前
        self.joint_targets[1] = phi4_ik              # 右后
        self.joint_targets[2] = phi4_ik              # 左前（对应左腿 phi4）
        self.joint_targets[3] = phi1_ik - math.pi    # 左后（对应左腿 phi1）

        return True

    def _joint_error(self, joint_pos):
        """计算关节角度与目标的最大绝对误差"""
        return max(abs(joint_pos[i] - self.joint_targets[i]) for i in range(4))

    def _get_leg_state(self, joint_pos, pitch, gyro_y):
        """用正运动学获取当前腿部状态"""
        self.vmc_r.vmc_calc_pos(
            phi1=joint_pos[0] + math.pi,
            phi4=joint_pos[1],
            pitch=pitch,
            gyro=gyro_y,
        )
        self.vmc_l.vmc_calc_pos(
            phi1=joint_pos[3] + math.pi,
            phi4=joint_pos[2],
            pitch=-pitch,
            gyro=-gyro_y,
        )

    def compute(self, joint_pos, pitch, gyro_y, body_x, body_vx):
        """
        分阶段计算控制输出

        返回:
            joint_torque: 关节力矩 [右前, 右后, 左前, 左后]
            wheel_torque: 轮子力矩（左右相同）
        """
        self.phase_tick += 1

        # ===== 阶段0：自由落体 =====
        if self.phase == PHASE_FREEFALL:
            if self.phase_tick >= self.freefall_ticks:
                self.phase = PHASE_LEG_CTRL
                self.phase_tick = 0
                # 重置所有PID积分，避免累积误差
                for pid in self.pid_joint:
                    pid.integral = 0
                print(f"[阶段切换] 落地完成 → 开始控制腿部")
            return [0.0] * 4, 0.0

        # ===== 阶段1：控制腿部到目标姿态 =====
        if self.phase == PHASE_LEG_CTRL:
            # 关节PID
            joint_torque = [0.0] * 4
            for i in range(4):
                joint_torque[i] = self.pid_joint[i].calc(joint_pos[i], self.joint_targets[i])

            # 判断是否腿部已到位（持续满足阈值）
            err = self._joint_error(joint_pos)
            if err < self.leg_settle_threshold and self.phase_tick > 250:
                self.phase = PHASE_BALANCE
                self.phase_tick = 0
                # 重置轮子PID积分
                self.pid_pitch.integral = 0
                self.pid_gyro.integral = 0
                self.pid_x.integral = 0
                self.pid_v.integral = 0
                # 记录当前位置作为平衡目标原点
                self.balance_x0 = body_x
                print(f"[阶段切换] 腿部到位 (err={err:.4f}) → 开始平衡控制")

            return joint_torque, 0.0

        # ===== 阶段2：全部控制，站立平衡 =====
        # 关节PID
        joint_torque = [0.0] * 4
        for i in range(4):
            joint_torque[i] = self.pid_joint[i].calc(joint_pos[i], self.joint_targets[i])

        # 平衡PID：pitch → 0, gyro → 0
        t_pitch = -self.pid_pitch.calc(pitch, 0.0)
        t_gyro  = -self.pid_gyro.calc(gyro_y, 0.0)

        # 位移和速度PID：x → x0, v → 0
        t_x = self.pid_x.calc(body_x, self.balance_x0)
        t_v = self.pid_v.calc(body_vx, 0.0)

        # 轮子力矩限幅 (MJCF ctrlrange = [-4, 4])
        wheel_torque = max(-4.0, min(4.0, t_pitch + t_gyro + t_x + t_v))

        return joint_torque, wheel_torque
