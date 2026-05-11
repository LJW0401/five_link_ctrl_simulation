import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from environment import *
from keyboard import *
from BalanceController import create_controller, CTRL_LQR, CTRL_PID, CTRL_MPC
from StateEstimator import StateEstimator, IMUData, MotorData
from CommandServer import CommandServer, apply_to_controller


def main():
    parser = argparse.ArgumentParser(description='Five-link leg-wheel robot simulation')
    parser.add_argument('--model', default='MJCF/env.xml',
                        help='MJCF entry file. e.g. MJCF/env.xml (STL mesh) or '
                             'MJCF_primitive/env.xml (cylinder/box primitives)')
    parser.add_argument('--fast', dest='realtime', action='store_false', default=True,
                        help='禁用 realtime 节流，让仿真按 CPU 全速跑（默认 1× realtime）')
    args = parser.parse_args()

    GBC486 = LegWheelRobot(args.model)
    i = 0
    t1 = 1   # 传感器读取周期 (ms)
    t2 = 4   # 控制计算周期 (ms)
    t3 = 20  # 打印周期 (ms)
    keyboard = KeyboardController()

    # 选择控制器: CTRL_LQR 或 CTRL_PID
    ctrl = create_controller(CTRL_MPC)
    # 监控用状态估计器（复用控制器的五连杆参数）
    leg_params = getattr(ctrl, 'leg_params', None)
    state = StateEstimator(leg_params)

    print(f"控制器: {type(ctrl).__name__} | 目标: L0={ctrl.L0_target:.3f}m")

    # 启动 UDP 指令服务器
    cmd_server = CommandServer(host="127.0.0.1", port=9000)
    cmd_server.start()

    # realtime 节流：每个 mj_step 推进 model.opt.timestep（默认 1 ms 仿真时间），
    # 用 perf_counter 精确补偿累计 wall-time 误差，避免单次 sleep 抖动累积
    # --fast 关闭节流，让 CPU 全速跑
    step_dt = GBC486.model.opt.timestep
    loop_start = time.perf_counter()
    print(f"时间模式: {'1× realtime' if args.realtime else '全速 (无节流)'}")

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

        if i % t3 == 0:
            cmd = keyboard.get_command()
            # print(
            #     f"θ={state.leg[0].theta:+.4f} dθ={state.leg[0].dTheta:+.3f} | "
            #     f"x={state.body.x:+.4f} dx={state.body.x_dot:+.3f} | "
            #     f"φ={state.body.phi:+.4f} dφ={state.body.phi_dot:+.3f} | "
            #     f"L0={state.leg[0].L0:.3f} whl={GBC486.wheel_torque[0]:+.2f}"
            # )

        # 节流到 1× realtime：基于"应该到达的总 wall-time"补偿，自然纠偏累计漂移
        # （如果某帧耗时超 step_dt，下一帧不会 sleep，自动追赶；不会出现累计提前）
        if args.realtime:
            target_wall = loop_start + i * step_dt
            remaining = target_wall - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)


if __name__ == '__main__':
    main()
