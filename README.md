# five_link_ctrl_simulation

基于 MuJoCo 的五连杆腿-轮机器人仿真平台（GBC486）。每条腿为五连杆并联机构，足端装驱动轮，支持在关节空间和虚拟腿空间（L0、φ0）之间双向映射。

提供三种平衡控制器可切换：**PID / LQR / MPC**。

---

## 运行环境

- Python 3.10
- Ubuntu 22.04

```bash
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt   # mujoco, numpy, pynput, scipy, matplotlib
```

## 启动仿真

```bash
python3 Simulation.py
```

在 `Simulation.py` 中切换控制器：

```python
from BalanceController import create_controller, CTRL_PID, CTRL_LQR, CTRL_MPC
ctrl = create_controller(CTRL_MPC)   # 或 CTRL_LQR / CTRL_PID
```

---

## 目录结构

| 文件 | 作用 |
|---|---|
| `Simulation.py` | 主循环：传感 1 ms / 控制 4 ms / 打印 20 ms |
| `environment.py` | MuJoCo 封装（读传感器、下发力矩） |
| `StateEstimator.py` | IMU + 电机反馈 → 机体状态 + 两腿状态 |
| `VMC.py` | 五连杆正/逆运动学、雅可比、力/力矩映射 |
| `Controller.py` | 基础 PID 类 |
| `BalanceController.py` | 控制器工厂 |
| `PIDController.py` | PID 级联：关节 PID + 位置外环 + pitch 内环 + yaw 差分 |
| `LQRController.py` | LQR 状态反馈（K 矩阵按 L0 线性插值）|
| `MPCController.py` | MPC 状态反馈（condensed QP + 投影梯度）|
| `calc_lqr_k.py` | 离线计算 LQR K 查找表，生成 `lqr_config.json` |
| `calc_mpc_config.py` | 离线计算 MPC 模型表与终端代价，生成 `mpc_config.json` |
| `MJCF/` | 原始 STL 机器人模型（`env.xml` 为入口） |
| `MJCF_primitive/` | 原始构型的基础几何替身模型 |
| `MJCF_rhombus/` | 菱形对称 5-bar 几何替身模型 |

---

## 模型构型

仓库中保留了 3 套 MuJoCo 模型，入口文件均为各目录下的 `env.xml`。

| 模型目录 | 构型 | 说明 |
|---|---|---|
| `MJCF/` | 原始 GBC486 STL 装配构型 | 直接加载 `base_link.STL`、`AG.STL`、`GH.STL`、`AB.STL` 等网格文件。每条腿保留原始五连杆布局，前后 hip 在车身上有偏置，上下支链存在交叉/折叠关系。 |
| `MJCF_primitive/` | 原始构型的几何替身 | 用 `box` / `cylinder` 等基础几何体替代 STL mesh，但保留与 `MJCF/robot.xml` 基本一致的 body 树、joint、actuator、sensor、site 和 equality 命名。适合调试动力学、碰撞和控制逻辑。 |
| `MJCF_rhombus/` | 对称菱形 5-bar 几何替身 | 当前分支新增的简化构型。每条腿由前后两条二连杆支链组成，foot/wheel 位于菱形下顶点；右腿拓扑为 `AG -> GH -> wheel_right` 与 `AB -> BE -> F`，通过 equality 在足端闭合。 |

三者的主要区别：

- `MJCF/`：最接近原始 CAD/URDF 的 STL 装配模型。
- `MJCF_primitive/`：不改变原始拓扑，只把可视和碰撞几何简化为基础体。
- `MJCF_rhombus/`：改变腿部几何为更规整的菱形 5-bar，参数为 `L1=L4=0.15 m`、`L2=L3=0.24 m`、hip 间距 `L5=0.10 m`，默认腿长 `L0=0.30 m`。

---

## 控制架构

所有控制器共用这条链：

```
传感器 ──► StateEstimator ──► 控制器核心（PID/LQR/MPC）
                                    │
                                    ├── 轮子力矩 T  ──► 左右轮（叠加 yaw PID 差分）
                                    │
                                    └── 髋部虚拟力矩 Tp ──► VMC 雅可比 ──► 关节力矩
                                                                 ▲
                                                腿长 PID + 重力前馈 (F0)
```

状态定义（LQR/MPC 共用 6 维）：
```
x = [theta, d_theta, x, d_x, phi, d_phi]
u = [T (轮力矩), Tp (髋部虚拟力矩)]
```

## 控制器对比

| 特性 | PID | LQR | MPC |
|---|---|---|---|
| 模型依赖 | 无 | 线性化模型 | 线性化模型 |
| 约束处理 | 无 | 事后截断 | **原生盒约束** |
| 时变模型 | 不相关 | 按 L0 插值 K | 按 L0 插值 (Ad, Bd, P) |
| 在线计算 | 极低 | 极低（矩阵乘）| 每步 ~0.5 ms（N=15 QP）|

---

## LQR 使用

生成 K 查找表：
```bash
python3 calc_lqr_k.py   # → lqr_config.json, k_fitting.png
```

参数在 `calc_lqr_k.py` 顶部：`DEFAULT_Q`, `DEFAULT_R`, `DEFAULT_L0_RANGE`。

## MPC 使用

生成模型与终端代价表，并画出 MPC 第一步等效增益 vs L0（与 LQR K 叠加对比）：
```bash
python3 calc_mpc_config.py   # → mpc_config.json, mpc_k_fitting.png
```

参数在 `calc_mpc_config.py` 顶部：
- `DEFAULT_Q`, `DEFAULT_R`：状态/输入代价
- `DEFAULT_DT`, `DEFAULT_N`：预测步长与 horizon
- `DEFAULT_U_MIN`, `DEFAULT_U_MAX`：输入盒约束 `[T, Tp]`

求解器：condensed QP + 投影梯度下降（warm start、30 次迭代、α=1/λmax）。无需 OSQP 等外部 QP 求解器。

---

## 约定

- 电机顺序：`[右前关节, 右后关节, 左前关节, 左后关节, 右轮, 左轮]`
- 坐标约定：`phi = -pitch`，`theta = π/2 - phi0 + pitch`（theta=0 时腿竖直向下）
- 代码注释中英混用
