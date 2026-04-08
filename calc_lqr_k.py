"""
LQR K 矩阵计算（纯数值方法）

用法: python calc_lqr_k.py
  → 计算多个腿长下的 K 矩阵，保存到 lqr_config.json

原理:
  1. 数值有限差分求 A, B 矩阵
  2. scipy Riccati 求解 LQR
  3. 结果存为 JSON 查找表
"""

import json
import numpy as np
from scipy.linalg import solve_continuous_are

CONFIG_FILE = "lqr_config.json"

# ========== 默认参数 ==========
DEFAULT_ROBOT_PARAMS = {
    "R":  0.088,             # 驱动轮半径 (m)
    "l":  0.0447,            # 机体质心距转轴距离 (m)
    "mw": 0.322,             # 单个驱动轮质量 (kg)
    "mp": 2.751,             # 单腿质量 (kg)（不含轮子）
    "M":  8.4,               # 机体质量 (kg)
    "IM": 0.112177,          # 机体绕pitch轴转动惯量 (kg·m²)
    "g":  9.81,
}

DEFAULT_LEG_PARAMS = {
    "l1": 0.215,             # 连杆1长度 (m)
    "l2": 0.258,             # 连杆2长度 (m)
    "l3": 0.258,             # 连杆3长度 (m)
    "l4": 0.215,             # 连杆4长度 (m)
    "l5": 0.0,               # A-E距离 (m)
}

DEFAULT_Q = [100.0, 1.0, 500.0, 100.0, 5000.0, 1.0]
DEFAULT_R = [250.0, 100.0]

DEFAULT_L0_RANGE = {"min": 0.10, "max": 0.40, "n_points": 30}


def dynamics(state, ctrl, leg_length, params):
    """非线性动力学方程"""
    p = params
    theta, d_theta, x, d_x, phi, d_phi = state
    T, Tp = ctrl

    L = leg_length / 2.0
    LM = leg_length / 2.0
    Iw = p["mw"] * p["R"] ** 2
    Ip = p["mp"] * ((L + LM) ** 2 + 0.05 ** 2) / 12.0

    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    R_v, l_v = p["R"], p["l"]
    mw_v, mp_v, M_v, g_v = p["mw"], p["mp"], p["M"], p["g"]

    c_x_ddth = R_v * (M_v * (L + LM) * ct + mp_v * L * ct)
    c_x_ddx  = Iw / R_v + mw_v * R_v + R_v * (M_v + mp_v)
    c_x_ddph = -R_v * M_v * l_v * cp
    rhs_x = (T + R_v * ((M_v * (L + LM) + mp_v * L) * d_theta ** 2 * st
                          - M_v * l_v * d_phi ** 2 * sp))

    N_nl = (M_v * (L + LM) + mp_v * L) * (-d_theta ** 2 * st) + M_v * l_v * d_phi ** 2 * sp
    NM_nl = M_v * (-(L + LM) * d_theta ** 2 * st + l_v * d_phi ** 2 * sp)

    P_base = (M_v + mp_v) * g_v + (-(M_v * (L + LM) + mp_v * L) * d_theta ** 2 * ct
                                     - M_v * l_v * d_phi ** 2 * cp)
    PM_base = M_v * g_v + M_v * (-(L + LM) * d_theta ** 2 * ct - l_v * d_phi ** 2 * cp)

    c_th_ddth = Ip + ((M_v * (L + LM) + mp_v * L) * st * L + M_v * (L + LM) * st * LM) * st \
                + ((M_v * (L + LM) + mp_v * L) * ct * L + M_v * (L + LM) * ct * LM) * ct
    c_th_ddx  = -((M_v + mp_v) * L + M_v * LM) * ct
    c_th_ddph = (M_v * l_v * sp * L + M_v * l_v * sp * LM) * st \
                + (M_v * l_v * cp * L + M_v * l_v * cp * LM) * ct

    rhs_th = ((P_base * L + PM_base * LM) * st - (N_nl * L + NM_nl * LM) * ct - T + Tp)

    c_ph_ddth = M_v * (L + LM) * ct * l_v * cp - M_v * (L + LM) * st * l_v * sp
    c_ph_ddx  = M_v * l_v * cp
    c_ph_ddph = p["IM"] + M_v * l_v ** 2

    rhs_ph = Tp + NM_nl * l_v * cp + PM_base * l_v * sp

    M_mat = np.array([
        [c_x_ddth,  c_x_ddx,  c_x_ddph],
        [c_th_ddth, c_th_ddx, c_th_ddph],
        [c_ph_ddth, c_ph_ddx, c_ph_ddph],
    ])
    rhs_vec = np.array([rhs_x, rhs_th, rhs_ph])

    dd = np.linalg.solve(M_mat, rhs_vec)
    return np.array([d_theta, dd[0], d_x, dd[1], d_phi, dd[2]])


def compute_AB(leg_length, params, eps=1e-6):
    """有限差分计算 A, B"""
    x0, u0 = np.zeros(6), np.zeros(2)
    f0 = dynamics(x0, u0, leg_length, params)

    A = np.zeros((6, 6))
    for i in range(6):
        xp = x0.copy(); xp[i] += eps
        A[:, i] = (dynamics(xp, u0, leg_length, params) - f0) / eps

    B = np.zeros((6, 2))
    for i in range(2):
        up = u0.copy(); up[i] += eps
        B[:, i] = (dynamics(x0, up, leg_length, params) - f0) / eps

    return A, B


def compute_k(leg_length, params=None, Q=None, R=None):
    """计算给定腿长的 LQR K 矩阵 (2×6)"""
    if params is None:
        params = DEFAULT_ROBOT_PARAMS
    if Q is None:
        Q = np.diag(DEFAULT_Q)
    if R is None:
        R = np.diag(DEFAULT_R)

    A, B = compute_AB(leg_length, params)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def generate_config(params=None, leg_params=None, Q=None, R=None, L0_range=None):
    """
    计算 K 矩阵查找表并保存为 JSON

    返回: config 字典
    """
    if params is None:
        params = DEFAULT_ROBOT_PARAMS
    if leg_params is None:
        leg_params = DEFAULT_LEG_PARAMS
    if Q is None:
        Q = DEFAULT_Q
    if R is None:
        R = DEFAULT_R
    if L0_range is None:
        L0_range = DEFAULT_L0_RANGE

    Q_mat = np.diag(Q)
    R_mat = np.diag(R)

    L0_values = np.linspace(L0_range["min"], L0_range["max"], L0_range["n_points"]).tolist()
    K_table = []

    print(f"计算 K: L0 ∈ [{L0_range['min']:.2f}, {L0_range['max']:.2f}], {L0_range['n_points']} 点...")
    for L0 in L0_values:
        K = compute_k(L0, params, Q_mat, R_mat)
        K_table.append(K.tolist())
    print("完成！")

    K_arr = np.array(K_table)  # (n, 2, 6)
    L0_arr = np.array(L0_values)

    # 绘制曲线
    plot_k_curve(L0_arr, K_arr)

    config = {
        "robot_params": params,
        "leg_params": leg_params,
        "Q": Q,
        "R": R,
        "L0_range": L0_range,
        "L0_values": L0_values,
        "K_table": K_table,
    }
    return config


def plot_k_curve(L0_arr, K_arr):
    """绘制 K 矩阵随 L0 变化的曲线"""
    import matplotlib.pyplot as plt

    state_names = ["theta", "d_theta", "x", "dx", "phi", "d_phi"]
    row_labels = ["T (wheel)", "Tp (hip)"]

    fig, axes = plt.subplots(2, 6, figsize=(20, 6))
    fig.suptitle("K matrix vs L0 (linear interpolation lookup table)", fontsize=14)

    for i in range(2):
        for j in range(6):
            ax = axes[i][j]
            ax.plot(L0_arr, K_arr[:, i, j], 'o-', markersize=3, linewidth=1.2)
            ax.set_title(f"k[{i}][{j}] ({state_names[j]})", fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
            if i == 1:
                ax.set_xlabel("L0 (m)", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("k_fitting.png", dpi=150)
    print("K矩阵曲线已保存到 k_fitting.png")


def save_config(config, filepath=CONFIG_FILE):
    """保存到 JSON"""
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"已保存到 {filepath}")


def load_config(filepath=CONFIG_FILE):
    """从 JSON 加载"""
    with open(filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("LQR K 矩阵计算")
    print("=" * 60)

    config = generate_config()
    save_config(config)

    # 验证
    L0_test = 0.25
    K_exact = compute_k(L0_test)
    print(f"\nL0={L0_test}m 精确 K:")
    print(f"  T  = {np.round(K_exact[0], 3)}")
    print(f"  Tp = {np.round(K_exact[1], 3)}")
