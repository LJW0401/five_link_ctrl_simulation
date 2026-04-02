import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from environment import *
from keyboard import *
from BalanceController import BalanceController, PHASE_FREEFALL, PHASE_LEG_CTRL, PHASE_BALANCE
from VMC import leg_VMC


PHASE_NAMES = {PHASE_FREEFALL: "落地", PHASE_LEG_CTRL: "调腿", PHASE_BALANCE: "平衡"}


def main():

    GBC486 = LegWheelRobot('MJCF/env.xml')
    i = 0
    t1 = 1   # 传感器读取周期 (ms)
    t2 = 4   # 控制计算周期 (ms)
    t3 = 20  # 打印周期 (ms)
    keyboard = KeyboardController()

    # 正运动学实例，用于监控
    vmc_r = leg_VMC()
    vmc_l = leg_VMC()

    # 初始化平衡控制器
    ctrl = BalanceController()
    if not ctrl.update_ik_targets():
        print("逆运动学无解，请检查目标参数！")
        return

    print(f"IK 目标关节角度: {[f'{a:.3f}' for a in ctrl.joint_targets]}")
    print(f"目标: L0={ctrl.L0_target:.3f}m  phi0={ctrl.phi0_target:.3f}rad")

    while True:
        i = i + 1

        GBC486.step()

        if i % t1 == 0:
            GBC486.sensor_read_data()

        if i % t2 == 0:
            joint_torque, wheel_torque = ctrl.compute(
                joint_pos=GBC486.joint_pos,
                pitch=GBC486.euler[1],
                gyro_y=GBC486.gyro[1],
                body_x=GBC486.body_x,
                body_vx=GBC486.body_vx,
            )

            GBC486.joint_torque = joint_torque
            GBC486.wheel_torque = [0.0, 0.0]  # 暂时不控制轮子
            GBC486.actuator_set_torque()

        if i % t3 == 0:
            cmd = keyboard.get_command()

            # 正运动学计算当前 L0, phi0
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

            phase_name = PHASE_NAMES.get(ctrl.phase, "?")
            print(
                f"[{phase_name}] "
                f"R: L0={vmc_r.L0:.4f} phi0={vmc_r.phi0:.3f} | "
                f"L: L0={vmc_l.L0:.4f} phi0={vmc_l.phi0:.3f} | "
                f"target: L0={ctrl.L0_target:.3f} phi0={ctrl.phi0_target:.3f} | "
                f"pitch={GBC486.euler[1]:.3f}"
            )


if __name__ == '__main__':
    main()
