"""
实验全局配置：仿真周期、IMU 噪声、控制器清单与输出路径。

与论文第四章保持一致：
  - 物理步长 / 传感周期 1 ms，控制周期 4 ms（250 Hz）
  - IMU 各通道叠加方差 1e-4（标准差 0.01）的零均值白噪声
  - 控制器：PID / LQR / MPC，共享同一状态-输入接口
所有路径相对于仓库根目录解析，便于从任意工作目录调用。
"""

import os

# ---- 路径 ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "MJCF", "env.xml")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

# ---- 多档周期（与 Simulation.py / 论文表 4.2 一致）----
DT_SIM = 0.001        # 物理 / 传感周期 (s)
CTRL_DECIM = 4        # 控制周期 = 4 * DT_SIM = 4 ms
DT_CTRL = DT_SIM * CTRL_DECIM

# ---- 预平衡热身（不记录）：让机器人从生成高度落地并稳定后再开始工况计时 ----
WARMUP_S = 3.0

# ---- IMU 白噪声（论文：σ² = 1e-4）----
IMU_NOISE_STD = 0.01
NOISE_SEED = 20260602  # 固定随机种子，保证可复现

# ---- 绘图起始时刻：裁掉前若干秒（只影响图，不影响 CSV 与指标）----
PLOT_START_S = 2.0

# ---- 控制器清单 ----
CONTROLLERS = ["pid", "lqr", "mpc"]
CONTROLLER_LABEL = {"pid": "PID", "lqr": "LQR", "mpc": "MPC"}
CONTROLLER_COLOR = {"pid": "#c0392b", "lqr": "#2766c8", "mpc": "#27895a"}

# ---- 执行器物理上限（与 MJCF ctrlrange 一致）----
WHEEL_TORQUE_LIMIT = 4.0   # 单轮力矩上限 (N·m)
SATURATION_FRAC = 0.98     # 判定饱和的阈值比例 (|T| >= frac * limit)
