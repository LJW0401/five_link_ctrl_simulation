"""
腿部与机体状态估计模块

坐标约定:
  phi0  — 虚拟腿极角（正运动学输出）
  phi   — 机体倾角 = -pitch
  theta — 摆杆相对竖直方向的夹角 = π/2 - phi0 - phi = π/2 - phi0 + pitch
  theta = 0 时腿竖直向下
  所有方向均从机体右侧看

传感器接口:
  IMUData  — 模拟机体 IMU（角度/角速度/加速度）
  MotorData — 模拟单个电机反馈（位置/速度/力矩）
  电机顺序: [右前关节, 右后关节, 左前关节, 左后关节, 右轮, 左轮]
"""

import math
from VMC import leg_VMC


class IMUData:
    """IMU 传感器数据"""
    __slots__ = ('r', 'p', 'y', 'dr', 'dp', 'dy', 'ax', 'ay', 'az')

    def __init__(self, r=0.0, p=0.0, y=0.0,
                 dr=0.0, dp=0.0, dy=0.0,
                 ax=0.0, ay=0.0, az=0.0):
        self.r = r      # roll  (rad)
        self.p = p      # pitch (rad)
        self.y = y      # yaw   (rad)
        self.dr = dr     # roll  角速度 (rad/s)
        self.dp = dp     # pitch 角速度 (rad/s)
        self.dy = dy     # yaw   角速度 (rad/s)
        self.ax = ax     # x 加速度 (m/s²)
        self.ay = ay     # y 加速度 (m/s²)
        self.az = az     # z 加速度 (m/s²)


class MotorData:
    """单个电机反馈数据"""
    __slots__ = ('pos', 'vel', 'tor')

    def __init__(self, pos=0.0, vel=0.0, tor=0.0):
        self.pos = pos   # 反馈位置 (rad)
        self.vel = vel   # 反馈速度 (rad/s)
        self.tor = tor   # 输出力矩 (Nm)


class LegState:
    """单条腿的状态"""
    def __init__(self):
        # 腿长
        self.L0 = 0.0
        self.dL0 = 0.0

        # 虚拟腿极角
        self.Phi0 = 0.0
        self.dPhi0 = 0.0

        # 摆杆夹角（相对竖直）
        self.Theta = 0.0
        self.dTheta = 0.0

        # 上一时刻值（用于数值微分）
        self._last_L0 = 0.0
        self._last_Phi0 = 0.0
        self._last_Theta = 0.0
        self._first = True


class BodyState:
    """机体状态"""
    def __init__(self):
        self.phi = 0.0       # 机体倾角 = -pitch
        self.phi_dot = 0.0   # 机体角速度 = -pitch_dot
        self.x = 0.0         # 位移
        self.x_dot = 0.0     # 速度


GRAVITY = 9.81
BODY_MASS = 8.4     # 机体质量 (kg)
WHEEL_MASS = 0.322   # 单轮质量 (kg)
WHEEL_RADIUS = 0.088 # 轮子半径 (m)


class StateEstimator:
    """双腿 + 机体状态估计器"""

    def __init__(self, leg_params=None):
        self.vmc_r = leg_VMC(leg_params)
        self.vmc_l = leg_VMC(leg_params)
        self.leg = [LegState(), LegState()]   # [右腿, 左腿]
        self.body = BodyState()

        # 轮式里程计
        self._odom_x = 0.0

    def update(self, imu, motors, dt=0.004):
        """
        更新全部状态

        参数:
            imu:    IMUData — 机体 IMU 数据
            motors: list[MotorData] — 6 个电机数据
                    顺序: [右前关节, 右后关节, 左前关节, 左后关节, 右轮, 左轮]
            dt:     控制周期 (s)
        """
        # --- 轮式里程计（直接用电机反馈速度） ---
        wheel_vel_r = motors[4].vel
        wheel_vel_l = motors[5].vel
        body_vx = (wheel_vel_r + wheel_vel_l) * 0.5 * WHEEL_RADIUS
        self._odom_x += body_vx * dt

        # --- 机体状态 ---
        self.body.phi = -imu.p
        self.body.phi_dot = -imu.dp
        self.body.x = self._odom_x
        self.body.x_dot = body_vx

        # --- 关节角度（从电机位置加偏移） ---
        joint_pos = [
            motors[0].pos,   # 右前
            motors[1].pos,   # 右后
            motors[2].pos,   # 左前
            motors[3].pos,   # 左后
        ]

        # --- 右腿正运动学 ---
        self.vmc_r.calc_forward_kinematics(
            phi1=-joint_pos[1] + math.pi,
            phi4=-joint_pos[0],
        )

        # print(f"VMC 右腿: phi1={self.vmc_r.phi1:.3f} rad, phi4={self.vmc_r.phi4:.3f} rad, ")
        # print(f"       L0={self.vmc_r.L0:.3f} m, phi0={self.vmc_r.phi0:.3f} rad, theta={self.vmc_r.theta:.3f} rad")
        # print(f"       dL0={self.vmc_r.d_L0:.3f} m/s, dPhi0={self.vmc_r.d_phi0:.3f} rad/s, dTheta={self.vmc_r.d_theta:.3f} rad/s")

        # --- 左腿正运动学 ---
        self.vmc_l.calc_forward_kinematics(
            phi1=joint_pos[3] + math.pi,
            phi4=joint_pos[2],
        )

        # print(f"VMC 左腿: phi1={self.vmc_l.phi1:.3f} rad, phi4={self.vmc_l.phi4:.3f} rad, ")
        # print(f"       L0={self.vmc_l.L0:.3f} m, phi0={self.vmc_l.phi0:.3f} rad, theta={self.vmc_l.theta:.3f} rad")
        # print(f"       dL0={self.vmc_l.d_L0:.3f} m/s, dPhi0={self.vmc_l.d_phi0:.3f} rad/s, dTheta={self.vmc_l.d_theta:.3f} rad/s")

        # --- 更新各腿状态 ---
        self._update_leg(0, self.vmc_r, self.body.phi, dt)  # 右腿
        self._update_leg(1, self.vmc_l, self.body.phi, dt)  # 左腿

        print("状态估计: ")
        print(f"         右腿 L0={self.leg[0].L0:.3f} m, phi0={self.leg[0].Phi0:.3f} rad, theta={self.leg[0].Theta:.3f} rad")
        print(f"         右腿 dL0={self.leg[0].dL0:.3f} m/s, dPhi0={self.leg[0].dPhi0:.3f} rad/s, dTheta={self.leg[0].dTheta:.3f} rad/s")
        print("")
        print(f"         左腿 L0={self.leg[1].L0:.3f} m, phi0={self.leg[1].Phi0:.3f} rad, theta={self.leg[1].Theta:.3f} rad")
        print(f"         左腿 dL0={self.leg[1].dL0:.3f} m/s, dPhi0={self.leg[1].dPhi0:.3f} rad/s, dTheta={self.leg[1].dTheta:.3f} rad/s")

    def _update_leg(self, idx, vmc, body_phi, dt):
        """更新单条腿的状态"""
        leg = self.leg[idx]

        # ===== 位置 =====
        leg.L0 = vmc.L0
        leg.Phi0 = vmc.Phi0
        leg.Theta = math.pi/2.0 + body_phi - leg.Phi0

        if leg._first:
            leg._last_L0 = leg.L0
            leg._last_Phi0 = leg.Phi0
            leg._last_Theta = leg.Theta
            leg._first = False

        # ===== 速度（数值微分） =====
        leg.dL0 = (leg.L0 - leg._last_L0) / dt
        leg.dPhi0 = (leg.Phi0 - leg._last_Phi0) / dt
        leg.dTheta = (leg.Theta - leg._last_Theta) / dt

        # ===== 保存上一时刻 =====
        leg._last_L0 = leg.L0
        leg._last_Phi0 = leg.Phi0
        leg._last_Theta = leg.Theta

    @staticmethod
    def gravity_feedforward(theta):
        """重力前馈: F_ff = m*g*cos(theta) / 2（每条腿分担一半体重）"""
        return BODY_MASS * GRAVITY * math.cos(theta) / 2.0
