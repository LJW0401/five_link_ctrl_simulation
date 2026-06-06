import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import argparse
from environment import *
from keyboard import *
from BalanceController import create_controller, CTRL_LQR, CTRL_PID, CTRL_MPC
from StateEstimator import StateEstimator, IMUData, MotorData
from CommandServer import CommandServer, apply_to_controller
from MonitorServer import MonitorServer


def parse_args():
    ap = argparse.ArgumentParser(description="五连杆机器人仿真")
    ap.add_argument("--monitor", action="store_true",
                    help="开启监控记录后端服务（网页界面 + CSV 导出）")
    ap.add_argument("--monitor-host", default="127.0.0.1", help="监控服务监听地址")
    ap.add_argument("--monitor-port", type=int, default=8000, help="监控服务端口")
    return ap.parse_args()


def main():

    args = parse_args()

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

    # 启动 UDP 指令服务器
    cmd_server = CommandServer(host="127.0.0.1", port=9000)
    cmd_server.start()

    # 按命令行开关启动监控记录后端服务
    monitor = None
    if args.monitor:
        monitor = MonitorServer(host=args.monitor_host, port=args.monitor_port)
        monitor.start()

    while True:
        i = i + 1

        GBC486.step()

        if i % t1 == 0:
            GBC486.sensor_read_data()

        if i % t2 == 0:
            # 应用 UDP 指令到控制器目标
            apply_to_controller(cmd_server.snapshot(), ctrl)

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
            GBC486.wheel_torque = wheel_torque
            GBC486.actuator_set_torque()

            # 推送监控样本（右腿六变量反馈 + T/Tp 输出）
            if monitor is not None:
                monitor.push({
                    "t": GBC486.data.time,
                    "theta": state.leg[0].Theta,
                    "d_theta": state.leg[0].dTheta,
                    "x": state.body.x,
                    "d_x": state.body.x_dot,
                    "phi": state.body.phi,
                    "d_phi": state.body.phi_dot,
                    "T": getattr(ctrl, "T", 0.0),
                    "Tp_r": getattr(ctrl, "Tp_r", 0.0),
                    "Tp_l": getattr(ctrl, "Tp_l", 0.0),
                })

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
