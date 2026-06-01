# body_move_kinematic.py
#
# 职责：独立的运动学演示仿真——直接写机体自由关节的位置，让机体沿 x 轴前后往复移动。
#       不调用 mj_step（不积分动力学、不受重力/碰撞影响），只用 mj_forward 刷新运动学，
#       机体悬于固定高度做纯位置回放。用于直观观察 rhombus 五连杆模型的形态与机体平移。
#
# 与主仿真 Simulation.py 互不影响：Simulation.py 仍用 MJCF/ 的力矩控制模型，
#       本脚本默认加载 MJCF_rhombus/env.xml（可改 MODEL_PATH 切换）。
#
# 依赖：mujoco、numpy。base 本体的 base_free 自由关节，qpos[0:3]=位置[x,y,z]，qpos[3:7]=四元数。
#
# 运行：python3 body_move_kinematic.py   （MuJoCo 查看器，ESC 退出）

import time
import math
import mujoco
import mujoco.viewer

MODEL_PATH = 'MJCF_rhombus/env.xml'  # 想看 STL 版改成 'MJCF/env.xml'

AMPLITUDE = 0.5   # 前后移动幅度 (m)，机体在 x0±AMPLITUDE 之间往复
PERIOD = 4.0      # 往复周期 (s)
DT = 0.01         # 画面刷新步长 (s)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 定位机体自由关节在 qpos 中的起始地址（前 3 个分量即 x/y/z 位置）
    base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'base_free')
    qadr = model.jnt_qposadr[base_jid]

    # 摆出初始构型，记录初始位置作为往复中心
    mujoco.mj_forward(model, data)
    x0 = data.qpos[qadr + 0]

    print(f"加载模型: {MODEL_PATH}")
    print(f"机体沿 x 在 [{x0 - AMPLITUDE:.2f}, {x0 + AMPLITUDE:.2f}] m 之间往复，周期 {PERIOD}s")
    print("纯运动学回放（不走动力学），ESC 退出")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_start = time.time()
        while viewer.is_running():
            t = time.time() - t_start

            # 直接控制机体位置：正弦往复实现前后移动
            data.qpos[qadr + 0] = x0 + AMPLITUDE * math.sin(2 * math.pi * t / PERIOD)

            # 只刷新运动学（前向运动学 + 站点/几何位姿），不积分动力学
            mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(DT)


if __name__ == '__main__':
    main()
