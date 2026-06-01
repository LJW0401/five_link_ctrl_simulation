"""
实验工况定义（对应论文第四章「实验工况设计」，yaw 阶跃已移除）。

每个工况用一个 Scenario 描述：
  - apply(t, ctrl)   每个控制周期按时间设置控制器目标（位置/速度/yaw/腿长/pitch）
  - disturbance(t)   返回施加在机体上的水平外力 (N)，无扰动时返回 0
  - duration         仿真时长 (s)；settle 为计算稳态指标时跳过的前段时长 (s)

工况与论文编号一一对应：
  1 平衡保持 / 2 位置阶跃 / 3 速度跟踪 / 4 yaw 阶跃 / 5 腿长动态 / 6 瞬态扰动
"""

import math


class Scenario:
    def __init__(self, key, index, title, duration, init_L0,
                 apply, disturbance=None, settle=2.0,
                 step_time=None, step_target=None, metric=None):
        self.key = key            # 文件名用短标识
        self.index = index        # 论文工况编号
        self.title = title        # 中文工况名
        self.duration = duration  # 总时长 (s)
        self.init_L0 = init_L0    # 初始腿长目标 (m)
        self.apply = apply        # apply(t, ctrl) -> None
        self._disturbance = disturbance
        self.settle = settle      # 稳态指标跳过的前段 (s)
        self.step_time = step_time  # 阶跃/扰动发生时刻 (s)，用于瞬态指标
        self.step_target = step_target  # 阶跃目标值（如位置阶跃终值 m）
        self.metric = metric or {}  # 该工况的「头条指标」描述

    def disturbance(self, t):
        if self._disturbance is None:
            return 0.0
        return self._disturbance(t)


# ---------- 各工况目标调度函数 ----------

def _apply_balance(t, ctrl):
    ctrl.L0_target = 0.20
    ctrl.x_target = 0.0
    ctrl.v_target = 0.0
    ctrl.yaw_target = 0.0


# 位置阶跃参数：t=20s 时阶跃到 5m
POS_STEP_TIME = 20.0
POS_STEP_TARGET = 5.0


def _apply_pos_step(t, ctrl):
    ctrl.L0_target = 0.20
    ctrl.v_target = 0.0
    ctrl.yaw_target = 0.0
    ctrl.x_target = POS_STEP_TARGET if t >= POS_STEP_TIME else 0.0


# 速度跟踪参数（与位置阶跃同样改法）：t=20s 时阶跃到 1 m/s
# （×10 的 3 m/s 会使 LQR 倒伏发散，取中间值 1 m/s 保留可比性）
VEL_STEP_TIME = 20.0
VEL_STEP_TARGET = 1.0


def _apply_vel_track(t, ctrl):
    ctrl.L0_target = 0.20
    ctrl.yaw_target = 0.0
    ctrl.v_target = VEL_STEP_TARGET if t >= VEL_STEP_TIME else 0.0


def _apply_leg_sine(t, ctrl):
    # L0 ∈ [0.15, 0.25] m（中心 0.20、幅值 0.05），0.1 Hz 正弦；1 s 后开始摆动，留出起始平衡段
    ctrl.x_target = 0.0
    ctrl.v_target = 0.0
    ctrl.yaw_target = 0.0
    if t >= 1.0:
        ctrl.L0_target = 0.20 + 0.05 * math.sin(2 * math.pi * 0.1 * (t - 1.0))
    else:
        ctrl.L0_target = 0.20


def _apply_disturb(t, ctrl):
    ctrl.L0_target = 0.20
    ctrl.x_target = 0.0
    ctrl.v_target = 0.0
    ctrl.yaw_target = 0.0


def _disturb_push(t):
    # 30 N 水平推力，作用区间 [2.0, 2.1] s（持续 0.1 s）
    return 30.0 if 2.0 <= t < 2.1 else 0.0


# ---------- 工况清单 ----------

SCENARIOS = [
    Scenario("balance", 1, "平衡保持", duration=20.0, init_L0=0.20,
             apply=_apply_balance, settle=3.0,
             metric={"name": "pitch RMS", "unit": "°"}),
    Scenario("pos_step", 2, "位置阶跃", duration=40.0, init_L0=0.20,
             apply=_apply_pos_step, settle=1.0,
             step_time=POS_STEP_TIME, step_target=POS_STEP_TARGET,
             metric={"name": "上升时间", "unit": "s"}),
    Scenario("vel_track", 3, "速度跟踪", duration=40.0, init_L0=0.20,
             apply=_apply_vel_track, settle=1.0,
             step_time=VEL_STEP_TIME, step_target=VEL_STEP_TARGET,
             metric={"name": "稳态速度误差", "unit": "m/s"}),
    # 工况 4（yaw 阶跃）已移除：yaw 由三控制器共用的同一 yaw PID 跟踪，
    # 在 PID/LQR/MPC 间完全相同；且转向几乎不扰动平衡（转向期 pitch 偏差与工况1雷同），
    # 不提供超出工况1的对比信息，故删去。工况编号保留原值（4 处留空）。
    Scenario("leg_sine", 5, "腿长动态变化", duration=22.0, init_L0=0.20,
             apply=_apply_leg_sine, settle=2.0,
             metric={"name": "pitch 振幅", "unit": "°"}),
    Scenario("disturb", 6, "瞬态扰动恢复", duration=6.0, init_L0=0.20,
             apply=_apply_disturb, disturbance=_disturb_push,
             settle=1.0, step_time=2.0,
             metric={"name": "pitch 峰值", "unit": "°"}),
]


def get_scenarios():
    return SCENARIOS
