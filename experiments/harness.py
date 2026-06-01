"""
无界面单次仿真执行器：给定控制器类型与工况，跑一次确定性仿真并返回时序数据。

与 Simulation.py 主循环等价，但：
  - 关闭 MuJoCo 可视化与实时 sleep（批跑加速、可复现）
  - 不走 UDP/键盘，工况目标由 scenario.apply(t, ctrl) 直接注入
  - IMU 读数叠加固定种子的零均值白噪声（仅作为控制器输入；
    指标统计使用 MuJoCo 真实状态，不含噪声）
  - 工况 6 通过 data.xfrc_applied 在机体上施加水平推力

返回 dict[str, np.ndarray]，每个键是一条按控制周期采样的时序。
"""

import os
import sys
import time
import math
import contextlib

import numpy as np
import mujoco

from environment import LegWheelRobot
from BalanceController import create_controller
from StateEstimator import IMUData, MotorData

from . import config


@contextlib.contextmanager
def _suppress_stdout():
    """屏蔽控制器/状态估计器内部的高频 print，避免污染批跑日志。"""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def _build_motors(robot):
    return [
        MotorData(pos=robot.joint_pos[0], vel=robot.joint_vel[0]),
        MotorData(pos=robot.joint_pos[1], vel=robot.joint_vel[1]),
        MotorData(pos=robot.joint_pos[2], vel=robot.joint_vel[2]),
        MotorData(pos=robot.joint_pos[3], vel=robot.joint_vel[3]),
        MotorData(pos=robot.right_wheel_pos, vel=robot.wheel_vel[0]),
        MotorData(pos=robot.left_wheel_pos, vel=robot.wheel_vel[1]),
    ]


def run_one(ctrl_type, scenario):
    """
    跑单个 (控制器, 工况)。

    参数:
        ctrl_type: "pid" / "lqr" / "mpc"
        scenario:  scenarios.Scenario 实例
    返回:
        dict[str, np.ndarray] 时序数据
    """
    rng = np.random.default_rng(config.NOISE_SEED)

    with _suppress_stdout():
        robot = LegWheelRobot(config.MODEL_PATH, viewer=False)
        ctrl = create_controller(ctrl_type)

    ctrl.L0_target = scenario.init_L0
    base_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "base")

    warmup_steps = int(round(config.WARMUP_S / config.DT_SIM))
    n_steps = int(round(scenario.duration / config.DT_SIM))

    # 时序缓存
    log = {k: [] for k in (
        "t", "pitch", "pitch_cmd", "yaw", "x", "x_true", "vx",
        "L0", "L0_target", "x_target", "v_target", "yaw_target",
        "T_right", "T_left", "Tp_r", "Tp_l", "solve_ms",
        # 控制器实际反馈的六维状态向量 x=(θ,θ̇,x,ẋ,φ,φ̇)（已减目标，即 u=-Kx 的 x）
        "s_theta", "s_dtheta", "s_x", "s_dx", "s_phi", "s_dphi",
    )}

    with _suppress_stdout():
        # --- 预平衡热身：施加工况初始目标，落地稳定后再正式计时（不记录）---
        for wstep in range(warmup_steps):
            robot.step()
            robot.sensor_read_data()
            if wstep % config.CTRL_DECIM == 0:
                scenario.apply(0.0, ctrl)
                noise = rng.normal(0.0, config.IMU_NOISE_STD, size=6)
                imu = IMUData(
                    r=robot.euler[0] + noise[0], p=robot.euler[1] + noise[1],
                    y=robot.euler[2] + noise[2], dr=robot.gyro[0] + noise[3],
                    dp=robot.gyro[1] + noise[4], dy=robot.gyro[2] + noise[5],
                )
                joint_torque, wheel_torque = ctrl.compute(imu, _build_motors(robot))
                robot.joint_torque = joint_torque
                robot.wheel_torque = wheel_torque
                robot.actuator_set_torque()

        # 热身结束：清零里程计与真实位移基准，使工况从「位置 0」起算
        # （热身落地捕获阶段累积的里程计漂移与初始目标无关，不应计入工况响应）
        ctrl.state._odom_x = 0.0
        x0_true = robot.data.xpos[base_id][0]

        for step in range(n_steps):
            t = step * config.DT_SIM

            # --- 工况 6：水平推力扰动（每个物理步设置/清零）---
            fx = scenario.disturbance(t)
            robot.data.xfrc_applied[base_id, :] = 0.0
            if fx != 0.0:
                robot.data.xfrc_applied[base_id, 0] = fx

            robot.step()
            robot.sensor_read_data()  # 传感周期 = 物理步长 = 1 ms

            if step % config.CTRL_DECIM == 0:
                scenario.apply(t, ctrl)

                # IMU 叠加白噪声（仅控制器输入）
                noise = rng.normal(0.0, config.IMU_NOISE_STD, size=6)
                imu = IMUData(
                    r=robot.euler[0] + noise[0],
                    p=robot.euler[1] + noise[1],
                    y=robot.euler[2] + noise[2],
                    dr=robot.gyro[0] + noise[3],
                    dp=robot.gyro[1] + noise[4],
                    dy=robot.gyro[2] + noise[5],
                )
                motors = _build_motors(robot)

                t0 = time.perf_counter()
                joint_torque, wheel_torque = ctrl.compute(imu, motors)
                solve_ms = (time.perf_counter() - t0) * 1e3

                robot.joint_torque = joint_torque
                robot.wheel_torque = wheel_torque
                robot.actuator_set_torque()

                # --- 记录真实状态（无噪声）---
                log["t"].append(t)
                log["pitch"].append(robot.euler[1])
                log["pitch_cmd"].append(getattr(ctrl, "pitch_ref", 0.0))
                log["yaw"].append(robot.euler[2])
                log["x"].append(ctrl.state.body.x)           # 里程计位置（控制器反馈量，x_target 同一坐标系）
                log["x_true"].append(robot.data.xpos[base_id][0] - x0_true)  # 真实位移（相对热身末）
                log["vx"].append(ctrl.state.body.x_dot)       # 里程计速度（控制器反馈量）
                log["L0"].append(ctrl.state.leg[0].L0)
                log["L0_target"].append(ctrl.L0_target)
                log["x_target"].append(getattr(ctrl, "x_target", 0.0))
                log["v_target"].append(getattr(ctrl, "v_target", 0.0))
                log["yaw_target"].append(getattr(ctrl, "yaw_target", 0.0))
                log["T_right"].append(wheel_torque[0])
                log["T_left"].append(wheel_torque[1])
                log["Tp_r"].append(getattr(ctrl, "Tp_r", 0.0))
                log["Tp_l"].append(getattr(ctrl, "Tp_l", 0.0))
                log["solve_ms"].append(solve_ms)

                # 六维状态反馈量（与 LQR/MPC 内部 u=-Kx 的 x 完全一致；右腿）
                leg0 = ctrl.state.leg[0]
                phi = ctrl.state.body.phi
                phi_dot = ctrl.state.body.phi_dot
                log["s_theta"].append(leg0.Theta)
                log["s_dtheta"].append(leg0.dTheta)
                log["s_x"].append(ctrl.state.body.x - getattr(ctrl, "x_target", 0.0))
                log["s_dx"].append(ctrl.state.body.x_dot - getattr(ctrl, "v_target", 0.0))
                log["s_phi"].append(-phi - getattr(ctrl, "pitch_target", 0.0))
                log["s_dphi"].append(-phi_dot)

    return {k: np.asarray(v, dtype=float) for k, v in log.items()}
