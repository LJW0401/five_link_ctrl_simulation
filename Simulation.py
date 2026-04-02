import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from environment import *
from keyboard import *
from BalanceController import create_controller, CTRL_LQR, CTRL_PID
from StateEstimator import StateEstimator


def main():

    GBC486 = LegWheelRobot('MJCF/env.xml')
    i = 0
    t1 = 1   # 传感器读取周期 (ms)
    t2 = 4   # 控制计算周期 (ms)
    t3 = 20  # 打印周期 (ms)
    keyboard = KeyboardController()

    # 选择控制器: CTRL_LQR 或 CTRL_PID
    ctrl = create_controller(CTRL_LQR)
    state = StateEstimator()

    print(f"控制器: {type(ctrl).__name__} | 目标: L0={ctrl.L0_target:.3f}m")

    while True:
        i = i + 1

        GBC486.step()

        if i % t1 == 0:
            GBC486.sensor_read_data()

        if i % t2 == 0:
            joint_torque, wheel_torque = ctrl.compute(
                joint_pos=GBC486.joint_pos,
                pitch=GBC486.euler[1],
                body_x=GBC486.body_x,
                body_vx=GBC486.body_vx,
                gyro_y=GBC486.gyro[1],
            )

            # 更新状态估计（用于监控）
            state.update(GBC486.joint_pos, GBC486.euler[1], GBC486.gyro[1],
                         GBC486.body_x, GBC486.body_vx)

            GBC486.joint_torque = joint_torque
            GBC486.wheel_torque = [wheel_torque, wheel_torque]
            GBC486.actuator_set_torque()

        if i % t3 == 0:
            cmd = keyboard.get_command()
            print(
                f"L0: R={state.leg[0].L0:.3f} L={state.leg[1].L0:.3f} | "
                f"θ: R={state.leg[0].theta:+.3f} L={state.leg[1].theta:+.3f} | "
                f"pitch={GBC486.euler[1]:+.4f} | "
                f"x={GBC486.body_x:+.4f} v={GBC486.body_vx:+.3f} | "
                f"whl={GBC486.wheel_torque[0]:+.2f}"
            )


if __name__ == '__main__':
    main()
