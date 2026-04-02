"""
腿部与机体状态估计模块
参考: Modular_Balancing_Robot_Controller/application/chassis/chassis_balance.c

坐标约定:
  phi0  — 虚拟腿极角（正运动学输出）
  phi   — 机体倾角 = -pitch
  theta — 摆杆相对竖直方向的夹角 = π/2 - phi0 - phi = π/2 - phi0 + pitch
  theta = 0 时腿竖直向下
"""

import math
from VMC import leg_VMC


class LegState:
    """单条腿的状态"""
    def __init__(self):
        # 腿长
        self.L0 = 0.0
        self.dL0 = 0.0
        self.ddL0 = 0.0

        # 虚拟腿极角
        self.phi0 = 0.0
        self.dPhi0 = 0.0
        self.ddPhi0 = 0.0

        # 摆杆夹角（相对竖直）
        self.theta = 0.0
        self.dTheta = 0.0
        self.ddTheta = 0.0

        # 上一时刻值（用于数值微分）
        self._last_L0 = 0.0
        self._last_dL0 = 0.0
        self._last_phi0 = 0.0
        self._last_dPhi0 = 0.0
        self._last_dTheta = 0.0
        self._first = True


class BodyState:
    """机体状态"""
    def __init__(self):
        self.phi = 0.0       # 机体倾角 = -pitch
        self.phi_dot = 0.0   # 机体角速度 = -pitch_dot
        self.x = 0.0         # 位移
        self.x_dot = 0.0     # 速度


GRAVITY = 9.81
BODY_MASS = 15.8     # 机体质量 (kg)
WHEEL_MASS = 0.322   # 单轮质量 (kg)


class StateEstimator:
    """双腿 + 机体状态估计器"""

    def __init__(self):
        self.vmc_r = leg_VMC()
        self.vmc_l = leg_VMC()
        self.leg = [LegState(), LegState()]   # [右腿, 左腿]
        self.body = BodyState()

    def update(self, joint_pos, pitch, gyro_y, body_x, body_vx, dt=0.004):
        """
        更新全部状态

        参数:
            joint_pos: [右前, 右后, 左前, 左后] 关节角度
            pitch:     俯仰角 (rad)
            gyro_y:    俯仰角速度 (rad/s)
            body_x:    机体位移 (m)
            body_vx:   机体速度 (m/s)
            dt:        控制周期 (s)
        """
        # --- 机体状态 ---
        self.body.phi = -pitch
        self.body.phi_dot = -gyro_y
        self.body.x = body_x
        self.body.x_dot = body_vx

        # --- 右腿正运动学 ---
        self.vmc_r.vmc_calc_pos(
            phi1=joint_pos[0] + math.pi,
            phi4=joint_pos[1],
            pitch=pitch,
            gyro=gyro_y,
            dt=dt,
        )

        # --- 左腿正运动学（pitch/gyro 取反） ---
        self.vmc_l.vmc_calc_pos(
            phi1=joint_pos[3] + math.pi,
            phi4=joint_pos[2],
            pitch=-pitch,
            gyro=-gyro_y,
            dt=dt,
        )

        # --- 更新各腿状态 ---
        self._update_leg(0, self.vmc_r, dt)  # 右腿
        self._update_leg(1, self.vmc_l, dt)  # 左腿

    def _update_leg(self, idx, vmc, dt):
        """更新单条腿的状态"""
        leg = self.leg[idx]

        # ===== 位置 =====
        leg.L0 = vmc.L0
        leg.phi0 = vmc.phi0
        # theta 直接使用 VMC 内部计算值（已处理左右腿镜像）
        leg.theta = vmc.theta

        if leg._first:
            leg._last_L0 = leg.L0
            leg._last_phi0 = leg.phi0
            leg._first = False

        # ===== 速度（数值微分） =====
        leg.dL0 = (leg.L0 - leg._last_L0) / dt
        leg.dPhi0 = (leg.phi0 - leg._last_phi0) / dt
        # dTheta 直接使用 VMC 内部值（已处理镜像）
        leg.dTheta = vmc.d_theta

        # ===== 加速度（数值微分） =====
        leg.ddL0 = (leg.dL0 - leg._last_dL0) / dt
        leg.ddPhi0 = (leg.dPhi0 - leg._last_dPhi0) / dt
        leg.ddTheta = (leg.dTheta - leg._last_dTheta) / dt

        # ===== 保存上一时刻 =====
        leg._last_L0 = leg.L0
        leg._last_dL0 = leg.dL0
        leg._last_phi0 = leg.phi0
        leg._last_dPhi0 = leg.dPhi0
        leg._last_dTheta = leg.dTheta

    @staticmethod
    def gravity_feedforward(theta):
        """重力前馈: F_ff = m*g*cos(theta) / 2（每条腿分担一半体重）"""
        return BODY_MASS * GRAVITY * math.cos(theta) / 2.0
