"""
LQR K 矩阵计算（纯数值方法）

用法: python calc_lqr_k.py
  → 计算多个腿长下的 K 矩阵，保存到 lqr_config.json

原理:
  1. 数值有限差分求 A, B 矩阵
  2. scipy Riccati 求解 LQR
  3. 对 K 的 12 个分量随 L0 分别做 3 阶多项式拟合，系数存为 JSON
"""

import json
import numpy as np
from scipy.linalg import solve_continuous_are

CONFIG_FILE = "lqr_config.json"

# ========== 默认参数 ==========
# 与 MJCF_rhombus/robot_rhombus.xml 对应：
#   - 单腿 4 根连杆质量 = 0.5(AG) + 1.0(GH) + 0.5(AB) + 1.0(BE) = 3.0 kg
#   - 上连杆 L1=L4=0.15 m，下连杆 L2=L3=0.24 m，hip 间距 L5=0.10 m
#   - base box 0.24×0.36×0.10 m, 质量 8.4 kg, COM 相对 hip 轴线沿 z 偏移 ≈ 0.05 m
#   - 关于 pitch (body y) 的转动惯量：I = M/12*(Lx²+Lz²) = 8.4/12*(0.24²+0.10²) ≈ 0.0473
DEFAULT_ROBOT_PARAMS = {
    "R":  0.088,             # 驱动轮半径 (m)
    "l": -0.05,              # 机体质心距 hip 转轴距离 (m)，对应 base_box 的 z 偏移
    "mw": 0.322,             # 单个驱动轮质量 (kg)
    "mp": 3.0,               # 单腿质量 (kg)（含 AG/GH/AB/BE 四根连杆，不含轮子）
    "M":  8.4,               # 机体质量 (kg)
    "IM": 0.0473,            # 机体绕 pitch 轴转动惯量 (kg·m²)
    "g":  9.81,
}

DEFAULT_LEG_PARAMS = {
    "l1": 0.15,              # 上连杆（rear, AG） (m)
    "l2": 0.24,              # 下连杆（rear, GH） (m)
    "l3": 0.24,              # 下连杆（front, BE） (m)
    "l4": 0.15,              # 上连杆（front, AB） (m)
    "l5": 0.10,              # A_rear ↔ A_front 沿 base x 的 hip 间距 (m)
}

#           [theta, d_theta, x,      d_x,   phi,    d_phi]
DEFAULT_Q = [50.0,   1.0,    100.0, 100.0, 2000.0, 10.0]
#           [T (wheel), Tp (hip)]
DEFAULT_R = [20.0, 1.0]

# 菱形 5 连杆腿长可行域：L1+L2=0.39 是几何上限，留余量；
# 下限避免接近完全折叠时的奇异。
DEFAULT_L0_RANGE = {"min": 0.12, "max": 0.36, "n_points": 30}

# K 分量随 L0 的多项式拟合阶数（与 matlab_k_table/get_k.m 一致）
POLY_ORDER = 3


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

    # 对 12 个分量分别做 POLY_ORDER 阶多项式拟合：
    #   K_poly_coef[i][j] = polyfit 系数（高次在前，长度 = POLY_ORDER+1）
    #   运行时 K[i][j](L0) = polyval(K_poly_coef[i][j], L0)
    K_poly_coef = [[None] * 6 for _ in range(2)]
    for i in range(2):
        for j in range(6):
            K_poly_coef[i][j] = np.polyfit(L0_arr, K_arr[:, i, j], POLY_ORDER).tolist()

    # 绘制曲线（散点 + 拟合）
    plot_k_curve(L0_arr, K_arr, K_poly_coef)

    config = {
        "robot_params": params,
        "leg_params": leg_params,
        "Q": Q,
        "R": R,
        "L0_range": L0_range,
        "poly_order": POLY_ORDER,
        "K_poly_coef": K_poly_coef,
    }
    return config


def plot_k_curve(L0_arr, K_arr, K_poly_coef):
    """绘制 K 矩阵随 L0 变化的曲线：逐点 LQR 散点 + 多项式拟合曲线"""
    import matplotlib.pyplot as plt

    state_names = ["theta", "d_theta", "x", "dx", "phi", "d_phi"]
    row_labels = ["T (wheel)", "Tp (hip)"]

    xf = np.linspace(L0_arr.min(), L0_arr.max(), 400)  # 加密网格画平滑拟合曲线

    fig, axes = plt.subplots(2, 6, figsize=(20, 6))
    fig.suptitle(f"K matrix vs L0 (order-{POLY_ORDER} polynomial fit)", fontsize=14)

    for i in range(2):
        for j in range(6):
            ax = axes[i][j]
            y_samp = K_arr[:, i, j]
            coef = K_poly_coef[i][j]
            y_hat = np.polyval(coef, L0_arr)   # 采样点处拟合值，用于算 R²

            # 拟合优度 R²
            ss_res = np.sum((y_samp - y_hat) ** 2)
            ss_tot = np.sum((y_samp - y_samp.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

            ax.scatter(L0_arr, y_samp, s=12, c='b', label='LQR sample')
            ax.plot(xf, np.polyval(coef, xf), 'r-', linewidth=1.4, label='poly fit')
            ax.set_title(f"k[{i}][{j}] ({state_names[j]})  R²={r2:.4f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
            if i == 1:
                ax.set_xlabel("L0 (m)", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
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
    L0_test = 0.30  # 菱形模型默认零位 hip→foot = 0.30 m
    K_exact = compute_k(L0_test)
    print(f"\nL0={L0_test}m 精确 K:")
    print(f"  T  = {np.round(K_exact[0], 3)}")
    print(f"  Tp = {np.round(K_exact[1], 3)}")
