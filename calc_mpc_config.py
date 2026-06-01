"""
MPC 预测模型配置生成

复用 calc_lqr_k.py 的连续线性化模型 (A, B)，
按 L0 离散化得到 (Ad, Bd)，并求解 DARE 得到终端代价 P。

产物：mpc_config.json
  - L0_values / Ad_table / Bd_table / P_table
  - Q, R, dt, N（控制器运行时可直接覆盖）
  - leg_params
"""

import json
import os
import numpy as np
from scipy.linalg import solve_discrete_are

from calc_lqr_k import (
    DEFAULT_ROBOT_PARAMS,
    DEFAULT_LEG_PARAMS,
    DEFAULT_L0_RANGE,
    compute_AB,
    compute_k as compute_lqr_k,
)
from MPCController import build_condensed

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "mpc_config.json")

# 状态代价   [theta, d_theta, x,    d_x,  phi,    d_phi]
DEFAULT_Q = [50.0, 1.0,      350.0, 100.0, 5000.0, 1.0]
# 输入代价 [T (wheel), Tp (hip)]
DEFAULT_R = [100.0, 1.0]

DEFAULT_DT = 0.02   # MPC 离散步长 (s)
DEFAULT_N = 15      # 预测 horizon 步数

# 输入盒约束（仅 wheel 力矩；Tp 由 VMC 下发，物理上通过 F0/雅可比间接限制）
DEFAULT_U_MIN = [-4.0, -10.0]
DEFAULT_U_MAX = [ 4.0,  10.0]


def discretize(A, B, dt):
    """一阶欧拉离散化：Ad = I + A·dt, Bd = B·dt"""
    n = A.shape[0]
    Ad = np.eye(n) + A * dt
    Bd = B * dt
    return Ad, Bd


def compute_terminal_P(Ad, Bd, Q, R):
    """解离散 Riccati，作为 MPC 终端代价 P"""
    return solve_discrete_are(Ad, Bd, Q, R)


def generate_config(robot_params=None, leg_params=None,
                    Q=None, R=None, L0_range=None,
                    dt=DEFAULT_DT, N=DEFAULT_N,
                    u_min=None, u_max=None):
    if robot_params is None:
        robot_params = DEFAULT_ROBOT_PARAMS
    if leg_params is None:
        leg_params = DEFAULT_LEG_PARAMS
    if Q is None:
        Q = DEFAULT_Q
    if R is None:
        R = DEFAULT_R
    if L0_range is None:
        L0_range = DEFAULT_L0_RANGE
    if u_min is None:
        u_min = DEFAULT_U_MIN
    if u_max is None:
        u_max = DEFAULT_U_MAX

    Q_mat = np.diag(Q)
    R_mat = np.diag(R)

    L0_values = np.linspace(L0_range["min"], L0_range["max"],
                            L0_range["n_points"]).tolist()

    Ad_table, Bd_table, P_table = [], [], []
    print(f"计算 MPC 模型: L0 ∈ [{L0_range['min']:.2f}, {L0_range['max']:.2f}], "
          f"{L0_range['n_points']} 点, dt={dt}s, N={N}")
    for L0 in L0_values:
        A, B = compute_AB(L0, robot_params)
        Ad, Bd = discretize(A, B, dt)
        P = compute_terminal_P(Ad, Bd, Q_mat, R_mat)
        Ad_table.append(Ad.tolist())
        Bd_table.append(Bd.tolist())
        P_table.append(P.tolist())
    print("完成！")

    config = {
        "robot_params": robot_params,
        "leg_params": leg_params,
        "Q": Q,
        "R": R,
        "dt": dt,
        "N": N,
        "u_min": u_min,
        "u_max": u_max,
        "L0_range": L0_range,
        "L0_values": L0_values,
        "Ad_table": Ad_table,
        "Bd_table": Bd_table,
        "P_table": P_table,
    }
    return config


def compute_mpc_gain(Ad, Bd, Q, R, P, N):
    """
    无约束 MPC 的等效第一步反馈增益:
        U* = -H⁻¹ Fx x₀,  K_mpc = [I₂, 0, …] · H⁻¹ · Fx  (2×6)
    用于与 LQR K 对比。
    """
    H, Fx = build_condensed(Ad, Bd, Q, R, P, N)
    nu = Bd.shape[1]
    U_gain = np.linalg.solve(H, Fx)  # (N·nu, nx)
    return U_gain[:nu, :]            # 取第一步 u_0 对应行


def plot_mpc_k_curve(L0_arr, Kmpc_arr, Klqr_arr=None, filepath="mpc_k_fitting.png"):
    """
    绘制 MPC 第一步反馈增益 vs L0；可选叠加 LQR K 作为对比。
    """
    import matplotlib.pyplot as plt

    state_names = ["theta", "d_theta", "x", "dx", "phi", "d_phi"]
    row_labels = ["T (wheel)", "Tp (hip)"]

    fig, axes = plt.subplots(2, 6, figsize=(20, 6))
    fig.suptitle("MPC first-step gain vs L0"
                 + (" (dashed = LQR K)" if Klqr_arr is not None else ""),
                 fontsize=14)

    for i in range(2):
        for j in range(6):
            ax = axes[i][j]
            ax.plot(L0_arr, Kmpc_arr[:, i, j], 'o-',
                    markersize=3, linewidth=1.2, label="MPC")
            if Klqr_arr is not None:
                ax.plot(L0_arr, Klqr_arr[:, i, j], '--',
                        linewidth=1.0, color='tab:red', label="LQR")
            ax.set_title(f"k[{i}][{j}] ({state_names[j]})", fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
            if i == 1:
                ax.set_xlabel("L0 (m)", fontsize=8)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0 and Klqr_arr is not None:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"MPC 增益曲线已保存到 {filepath}")


def save_config(config, filepath=CONFIG_FILE):
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"已保存到 {filepath}")


def load_config(filepath=CONFIG_FILE):
    with open(filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("MPC 模型 / 终端代价生成")
    print("=" * 60)
    config = generate_config()
    save_config(config)

    # ========== 绘图：MPC 等效第一步增益 vs L0（与 LQR 对比） ==========
    Q_mat = np.diag(config["Q"])
    R_mat = np.diag(config["R"])
    N = config["N"]
    L0_arr = np.array(config["L0_values"])
    Ad_arr = np.array(config["Ad_table"])
    Bd_arr = np.array(config["Bd_table"])
    P_arr  = np.array(config["P_table"])

    Kmpc_list, Klqr_list = [], []
    for i, L0 in enumerate(L0_arr):
        Kmpc = compute_mpc_gain(Ad_arr[i], Bd_arr[i], Q_mat, R_mat, P_arr[i], N)
        Klqr = compute_lqr_k(L0, config["robot_params"], Q_mat, R_mat)
        Kmpc_list.append(Kmpc)
        Klqr_list.append(Klqr)

    plot_mpc_k_curve(L0_arr,
                     np.array(Kmpc_list),
                     np.array(Klqr_list))
