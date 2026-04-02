import mujoco
import mujoco.viewer
import numpy as np
import time
from environment import *
from VMC import *
from keyboard import *
import math
from Controller import *

def main():

    GBC486 = LegWheelRobot('MJCF/env.xml')
    i = 0
    t1 = 1   # 传感器读取周期 (ms)
    t2 = 4   # 控制计算周期 (ms)
    t3 = 20  # 打印周期 (ms)
    vmc_r = leg_VMC()
    vmc_l = leg_VMC()
    keyboard = KeyboardController()

    # ========== 目标值 ==========
    L0_target = 0.2         # 目标腿长 (m)

    # ========== 腿部控制 PID ==========
    # F0：控制腿长 L0
    pid_L0_r = PID(p=1000.0, i=20.0, d=60.0, integral_limit=50.0, output_limit=300.0)
    pid_L0_l = PID(p=1000.0, i=20.0, d=60.0, integral_limit=50.0, output_limit=300.0)
    # Tp：控制 theta → 0（腿竖直 = 机体尽量水平）
    pid_theta_r = PID(p=30.0, i=5.0, d=8.0, integral_limit=50.0, output_limit=20.0)
    pid_theta_l = PID(p=30.0, i=5.0, d=8.0, integral_limit=50.0, output_limit=20.0)

    # 重力前馈
    gravity_ff = 15.8 * 9.81 / 2.0

    # ========== 轮子控制 PID ==========
    # 位移 → 0，速度 → 0（使用机体真实位置）
    pid_x = PID(p=5.0, i=0.1, d=0.0, integral_limit=20.0, output_limit=3.0)
    pid_v = PID(p=8.0, i=0.0, d=0.0, integral_limit=10.0, output_limit=4.0)

    while True:
        i = i + 1

        GBC486.step()

        if i % t1 == 0:
            GBC486.sensor_read_data()

        if i % t2 == 0:
            # --- 腿部 VMC ---
            vmc_r.vmc_calc_pos(
                phi1=GBC486.joint_pos[0] + math.pi,
                phi4=GBC486.joint_pos[1],
                pitch=GBC486.euler[1],
                gyro=GBC486.gyro[1],
            )
            vmc_l.vmc_calc_pos(
                phi1=GBC486.joint_pos[3] + math.pi,
                phi4=GBC486.joint_pos[2],
                pitch=-GBC486.euler[1],
                gyro=-GBC486.gyro[1],
            )

            # F0：控制腿长 + 重力前馈
            vmc_r.F0 = pid_L0_r.calc(vmc_r.L0, L0_target) + gravity_ff
            vmc_l.F0 = pid_L0_l.calc(vmc_l.L0, L0_target) + gravity_ff
            # Tp：控制 theta → 0，保持腿竖直
            vmc_r.Tp = pid_theta_r.calc(vmc_r.theta, 0.0)
            vmc_l.Tp = pid_theta_l.calc(vmc_l.theta, 0.0)

            vmc_r.vmc_calc_torque()
            vmc_l.vmc_calc_torque()

            # --- 轮子控制：位置、速度归零 ---
            w_pos = pid_x.calc(GBC486.body_x, 0.0)
            w_vel = pid_v.calc(GBC486.body_vx, 0.0)
            w_torque = max(-4.0, min(4.0, w_pos + w_vel))

            GBC486.wheel_torque = [w_torque, w_torque]
            GBC486.joint_torque = [
                vmc_r.torque_set[1], vmc_r.torque_set[0],
                vmc_l.torque_set[0], vmc_l.torque_set[1],
            ]
            GBC486.actuator_set_torque()

        if i % t3 == 0:
            cmd = keyboard.get_command()
            print(
                f"L0: R={vmc_r.L0:.3f} L={vmc_l.L0:.3f} | "
                f"theta: R={vmc_r.theta:.3f} L={vmc_l.theta:.3f} | "
                f"x={GBC486.body_x:.4f} v={GBC486.body_vx:.4f} pitch={GBC486.euler[1]:.3f}"
            )


if __name__ == '__main__':
    main()
