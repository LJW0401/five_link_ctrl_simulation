import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from environment import *
from keyboard import *
from BalanceController import create_controller, CTRL_LQR, CTRL_PID
from StateEstimator import StateEstimator, IMUData, MotorData


def main():

    GBC486 = LegWheelRobot('MJCF/env.xml')
    i = 0
    t1 = 1   # 传感器读取周期 (ms)
    t2 = 4   # 控制计算周期 (ms)
    t3 = 20  # 打印周期 (ms)
    keyboard = KeyboardController()

    # 选择控制器: CTRL_LQR 或 CTRL_PID
    ctrl = create_controller(CTRL_LQR)
    # 监控用状态估计器（复用控制器的五连杆参数）
    leg_params = getattr(ctrl, 'leg_params', None)
    state = StateEstimator(leg_params)

    print(f"控制器: {type(ctrl).__name__} | 目标: L0={ctrl.L0_target:.3f}m")

    while True:
        i = i + 1

        GBC486.step()

        if i % t1 == 0:
            GBC486.sensor_read_data()

        if i % t2 == 0:
            # 构造 IMU 数据
            imu = IMUData(
                r=GBC486.euler[0], p=GBC486.euler[1], y=GBC486.euler[2],
                dr=GBC486.gyro[0], dp=GBC486.gyro[1], dy=GBC486.gyro[2],
            )

            # 构造电机数据: [右前关节, 右后关节, 左前关节, 左后关节, 右轮, 左轮]
            motors = [
                MotorData(pos=GBC486.joint_pos[0], vel=GBC486.joint_vel[0]),
                MotorData(pos=GBC486.joint_pos[1], vel=GBC486.joint_vel[1]),
                MotorData(pos=GBC486.joint_pos[2], vel=GBC486.joint_vel[2]),
                MotorData(pos=GBC486.joint_pos[3], vel=GBC486.joint_vel[3]),
                MotorData(pos=GBC486.right_wheel_pos, vel=GBC486.wheel_vel[0]),
                MotorData(pos=GBC486.left_wheel_pos, vel=GBC486.wheel_vel[1]),
            ]


            # 更新状态估计（用于监控）
            state.update(imu, motors)
            
            # 计算控制输出
            joint_torque, wheel_torque = ctrl.compute(imu, motors)

            GBC486.joint_torque = joint_torque
            GBC486.wheel_torque = [wheel_torque, wheel_torque]
            GBC486.actuator_set_torque()

        if i % t3 == 0:
            cmd = keyboard.get_command()
            # print(
            #     f"θ={state.leg[0].theta:+.4f} dθ={state.leg[0].dTheta:+.3f} | "
            #     f"x={state.body.x:+.4f} dx={state.body.x_dot:+.3f} | "
            #     f"φ={state.body.phi:+.4f} dφ={state.body.phi_dot:+.3f} | "
            #     f"L0={state.leg[0].L0:.3f} whl={GBC486.wheel_torque[0]:+.2f}"
            # )


if __name__ == '__main__':
    main()
