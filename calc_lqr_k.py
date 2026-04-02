"""
LQR K 矩阵计算（纯数值方法，无需 sympy）

用法: python calc_lqr_k.py

原理:
  1. 数值有限差分求 A, B 矩阵（对非线性动力学方程在平衡点线性化）
  2. scipy LQR 求解
  3. numpy 多项式拟合
"""

import numpy as np
from scipy.linalg import solve_continuous_are


# ========== 机器人物理参数 ==========
class RobotParams:
    R = 0.088             # 驱动轮半径 (m)
    l = 0.03              # 机体质心距转轴距离 (m)
    mw = 0.322 * 2        # 驱动轮质量 (kg)
    mp = 1.8              # 摆杆（腿）质量 (kg)
    M = 8.4               # 机体质量 (kg)
    IM = 249032349.82e-9   # 机体绕质心转动惯量 (kg·m²)
    g = 9.81


# ========== Q/R 权重 ==========
Q_WEIGHTS = np.diag([100.0, 1.0, 500.0, 100.0, 5000.0, 1.0])
R_WEIGHTS = np.diag([220.0, 70.0])


def dynamics(state, ctrl, leg_length):
    """
    非线性动力学方程: state_dot = f(state, ctrl)

    状态: [theta, d_theta, x, d_x, phi, d_phi]
    控制: [T, Tp]

    求解联立方程得到 dd_theta, dd_x, dd_phi，
    返回 [d_theta, dd_theta, d_x, dd_x, d_phi, dd_phi]
    """
    p = RobotParams
    theta, d_theta, x, d_x, phi, d_phi = state
    T, Tp = ctrl

    L = leg_length / 2.0
    LM = leg_length / 2.0
    Iw = p.mw * p.R ** 2
    Ip = p.mp * ((L + LM) ** 2 + 0.05 ** 2) / 12.0

    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    # 构造 3×3 线性方程 M_mat * [dd_theta, dd_x, dd_phi]^T = rhs
    # 从三个运动方程整理得到

    # eqn1: (Iw/R + mw*R)*dd_x + R*(M*(L+LM)+mp*L)*ct*dd_theta - R*M*l*cp*dd_phi = T + nonlinear
    # eqn2: Ip*dd_theta + (M*(L+LM)^2+mp*L^2)*... = ...
    # eqn3: IM*dd_phi + ... = ...

    # 为避免手动推导容易出错，用 3x3 质量矩阵的方式
    # 将 N, NM, P, PM 中的 dd_x, dd_theta, dd_phi 项分离出来

    # dd_x 的方程系数
    c_x_ddth = p.R * (p.M * (L + LM) * ct + p.mp * L * ct)
    c_x_ddx  = Iw / p.R + p.mw * p.R + p.R * (p.M + p.mp)
    c_x_ddph = -p.R * p.M * p.l * cp
    rhs_x = (T + p.R * ((p.M * (L + LM) + p.mp * L) * d_theta ** 2 * st
                          - p.M * p.l * d_phi ** 2 * sp))

    # N, NM 不含 dd 项的非线性部分
    N_nl = ((p.M + p.mp) * 0  # dd_x 项移到左边
            + (p.M * (L + LM) + p.mp * L) * (-d_theta ** 2 * st)
            + p.M * (p.l * d_phi ** 2 * sp))
    NM_nl = p.M * (-(L + LM) * d_theta ** 2 * st + p.l * d_phi ** 2 * sp)

    # P, PM 不含 dd 项的部分
    P_base = (p.M + p.mp) * p.g + (-(p.M * (L + LM) + p.mp * L) * d_theta ** 2 * ct
                                     - p.M * p.l * d_phi ** 2 * cp)
    PM_base = p.M * p.g + p.M * (-(L + LM) * d_theta ** 2 * ct - p.l * d_phi ** 2 * cp)

    # dd_theta 方程: Ip*dd_theta = (P*L+PM*LM)*st - (N*L+NM*LM)*ct - T + Tp
    # P 中含 dd_theta 项: -(M*(L+LM)+mp*L)*dd_theta*st, PM 中: -M*(L+LM)*dd_theta*st
    # N 中含 dd_theta 项: (M*(L+LM)+mp*L)*dd_theta*ct, NM 中: M*(L+LM)*dd_theta*ct
    # P 中含 dd_phi 项: -M*l*dd_phi*sp, PM: -M*l*dd_phi*sp
    # N 中含 dd_phi 项: -M*l*dd_phi*cp, NM: -M*l*dd_phi*cp
    # N 中含 dd_x 项: (M+mp)*dd_x, NM: M*dd_x

    c_th_ddth = Ip + ((p.M * (L + LM) + p.mp * L) * st * L + p.M * (L + LM) * st * LM) * st \
                + ((p.M * (L + LM) + p.mp * L) * ct * L + p.M * (L + LM) * ct * LM) * ct
    c_th_ddx  = -((p.M + p.mp) * L + p.M * LM) * ct
    c_th_ddph_P = (p.M * p.l * sp * L + p.M * p.l * sp * LM) * st
    c_th_ddph_N = (p.M * p.l * cp * L + p.M * p.l * cp * LM) * ct
    c_th_ddph = c_th_ddph_P + c_th_ddph_N

    rhs_th = ((P_base * L + PM_base * LM) * st
              - (N_nl * L + NM_nl * LM) * ct
              - T + Tp)

    # dd_phi 方程: IM*dd_phi = Tp + NM*l*cp + PM*l*sp
    # NM 含 dd_x: M*dd_x, dd_theta: M*(L+LM)*ct, dd_phi: -M*l*cp
    # PM 含 dd_theta: -M*(L+LM)*st, dd_phi: -M*l*sp

    c_ph_ddth = p.M * (L + LM) * ct * p.l * cp - p.M * (L + LM) * st * p.l * sp
    c_ph_ddx  = p.M * p.l * cp
    c_ph_ddph = p.IM + p.M * p.l ** 2 * cp ** 2 + p.M * p.l ** 2 * sp ** 2

    rhs_ph = Tp + NM_nl * p.l * cp + PM_base * p.l * sp

    # 求解 [dd_theta, dd_x, dd_phi]
    M_mat = np.array([
        [c_x_ddth,  c_x_ddx,  c_x_ddph],
        [c_th_ddth, c_th_ddx,  c_th_ddph],
        [c_ph_ddth, c_ph_ddx,  c_ph_ddph],
    ])
    rhs_vec = np.array([rhs_x, rhs_th, rhs_ph])

    dd = np.linalg.solve(M_mat, rhs_vec)
    dd_theta_val, dd_x_val, dd_phi_val = dd

    return np.array([d_theta, dd_theta_val, d_x, dd_x_val, d_phi, dd_phi_val])


def compute_AB(leg_length, eps=1e-6):
    """用有限差分计算 A, B 矩阵"""
    x0 = np.zeros(6)
    u0 = np.zeros(2)
    f0 = dynamics(x0, u0, leg_length)

    A = np.zeros((6, 6))
    for i in range(6):
        x_plus = x0.copy()
        x_plus[i] += eps
        A[:, i] = (dynamics(x_plus, u0, leg_length) - f0) / eps

    B = np.zeros((6, 2))
    for i in range(2):
        u_plus = u0.copy()
        u_plus[i] += eps
        B[:, i] = (dynamics(x0, u_plus, leg_length) - f0) / eps

    return A, B


def compute_k(leg_length, Q=None, R=None):
    """计算给定腿长的 LQR K 矩阵 (2×6)"""
    if Q is None:
        Q = Q_WEIGHTS
    if R is None:
        R = R_WEIGHTS

    A, B = compute_AB(leg_length)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def fit_k_polynomials(L0_min=0.10, L0_max=0.40, n_points=30, poly_order=3):
    """对多个腿长计算 K 并拟合多项式"""
    L0_values = np.linspace(L0_min, L0_max, n_points)
    K_values = np.zeros((n_points, 2, 6))

    print(f"计算 K: L0 ∈ [{L0_min:.2f}, {L0_max:.2f}], {n_points} 点...")
    for idx, L0 in enumerate(L0_values):
        K_values[idx] = compute_k(L0)
    print("完成！")

    coeffs = [[None] * 6 for _ in range(2)]
    for i in range(2):
        for j in range(6):
            c = np.polyfit(L0_values, K_values[:, i, j], poly_order)
            coeffs[i][j] = c.tolist()

    return coeffs, L0_values, K_values


def print_python_code(coeffs):
    """输出可粘贴到 LQRController.py 的代码"""
    order = len(coeffs[0][0]) - 1
    labels = ["轮子力矩 T", "髋关节力矩 Tp"]
    state_names = ["theta", "d_theta", "x", "d_x", "phi", "d_phi"]
    print(f"\ndef get_k(L0):  # 多项式阶数: {order}")
    print("    k = [[0.0] * 6 for _ in range(2)]")
    for i in range(2):
        print(f"\n    # {labels[i]}")
        for j in range(6):
            c = coeffs[i][j]
            terms = [f"{c[n]:>12.4f} * L0**{order-n}" if order - n > 1
                     else (f"{c[n]:>12.4f} * L0" if order - n == 1
                           else f"{c[n]:>12.4f}")
                     for n in range(len(c))]
            print(f"    k[{i}][{j}] = {' + '.join(terms)}  # {state_names[j]}")
    print("\n    return k")


if __name__ == "__main__":
    p = RobotParams
    print("=" * 60)
    print("LQR K 矩阵计算（数值方法）")
    print("=" * 60)
    print(f"  R={p.R}m, mw={p.mw}kg, mp={p.mp}kg, M={p.M}kg")
    print(f"  Q = diag{list(np.diag(Q_WEIGHTS))}")
    print(f"  R = diag{list(np.diag(R_WEIGHTS))}")

    coeffs, L0_values, K_values = fit_k_polynomials(n_points=50, poly_order=5)
    print_python_code(coeffs)

    # 验证
    L0_test = 0.25
    K_exact = compute_k(L0_test)
    t1, t2, t3 = L0_test, L0_test ** 2, L0_test ** 3
    K_poly = np.zeros((2, 6))
    for i in range(2):
        for j in range(6):
            c = coeffs[i][j]
            K_poly[i, j] = c[0] * t3 + c[1] * t2 + c[2] * t1 + c[3]

    print(f"\n验证 L0={L0_test}m:")
    print(f"  最大绝对误差 = {np.max(np.abs(K_poly - K_exact)):.6f}")
    print(f"  最大相对误差 = {np.max(np.abs((K_poly - K_exact) / (np.abs(K_exact) + 1e-10))):.4%}")
