"""
MPC 状态反馈控制器（condensed form + 投影梯度 QP 求解）

状态向量 x[6]: [theta, d_theta, x, d_x, phi, d_phi]
控制输入 u[2]: [T (轮子力矩), Tp (髋关节虚拟力矩)]

流程：
  1. 启动时从 mpc_config.json 加载按 L0 参数化的 (Ad, Bd, P)
  2. 每步按当前 L0 线性插值得到 (Ad, Bd, P)
  3. 构造 condensed QP: min 0.5 Uᵀ H U + gᵀ U,  s.t. u_min ≤ u_k ≤ u_max
  4. 初值用 warm start，投影梯度迭代若干步
  5. 取 u_0 下发：T → 轮子力矩，Tp → VMC 髋部力
"""

import os
import json
import numpy as np

from StateEstimator import StateEstimator

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "mpc_config.json")


class MPCModelTable:
    """按 L0 线性插值 (Ad, Bd, P)"""

    def __init__(self, L0_values, Ad_table, Bd_table, P_table):
        self.L0 = np.asarray(L0_values, dtype=float)
        self.Ad = np.asarray(Ad_table, dtype=float)  # (n,6,6)
        self.Bd = np.asarray(Bd_table, dtype=float)  # (n,6,2)
        self.P  = np.asarray(P_table,  dtype=float)  # (n,6,6)
        self.n = len(L0_values)
        self.L0_min = self.L0[0]
        self.L0_max = self.L0[-1]
        self.step = (self.L0_max - self.L0_min) / (self.n - 1)

    def get(self, L0):
        if L0 <= self.L0_min:
            return self.Ad[0], self.Bd[0], self.P[0]
        if L0 >= self.L0_max:
            return self.Ad[-1], self.Bd[-1], self.P[-1]
        idx = int((L0 - self.L0_min) / self.step)
        idx = min(idx, self.n - 2)
        t = (L0 - self.L0[idx]) / (self.L0[idx + 1] - self.L0[idx])
        Ad = (1 - t) * self.Ad[idx] + t * self.Ad[idx + 1]
        Bd = (1 - t) * self.Bd[idx] + t * self.Bd[idx + 1]
        P  = (1 - t) * self.P[idx]  + t * self.P[idx + 1]
        return Ad, Bd, P


def build_condensed(Ad, Bd, Q, R, P, N):
    """
    构造 condensed QP 矩阵。
        X = Sx · x0 + Su · U
    其中 X = [x_1;…;x_N]，U = [u_0;…;u_{N-1}]
    代价 J(U) = 0.5 Uᵀ H U + (Fx · x0)ᵀ U + const
    """
    nx, nu = Bd.shape

    Sx = np.zeros((N * nx, nx))
    Su = np.zeros((N * nx, N * nu))

    A_pow = np.eye(nx)
    for k in range(N):
        A_pow = Ad @ A_pow            # A^(k+1)
        Sx[k * nx:(k + 1) * nx, :] = A_pow

    # Su: 下三角块矩阵，Su[k,j] = Ad^(k-j) · Bd   for j ≤ k
    for j in range(N):
        M = Bd.copy()
        for k in range(j, N):
            Su[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = M
            M = Ad @ M

    # Q_bar = diag(Q,Q,...,Q,P)（N 块，最后一块是 P）
    Q_bar = np.zeros((N * nx, N * nx))
    for k in range(N - 1):
        Q_bar[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = Q
    Q_bar[(N - 1) * nx:N * nx, (N - 1) * nx:N * nx] = P

    R_bar = np.zeros((N * nu, N * nu))
    for k in range(N):
        R_bar[k * nu:(k + 1) * nu, k * nu:(k + 1) * nu] = R

    H = 2.0 * (Su.T @ Q_bar @ Su + R_bar)
    Fx = 2.0 * Su.T @ Q_bar @ Sx
    # 对称化（数值稳定）
    H = 0.5 * (H + H.T)
    return H, Fx


def solve_qp_pgd(H, g, u_min, u_max, U0, n_iter=30):
    """
    投影梯度下降：min 0.5 Uᵀ H U + gᵀ U,  s.t. U_min ≤ U ≤ U_max
    U_min/U_max 已为完整长度向量。
    返回 U。
    """
    # 先尝试无约束闭式解；若已在可行域内直接返回
    try:
        U_free = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        U_free = U0.copy()

    if np.all(U_free >= u_min) and np.all(U_free <= u_max):
        return U_free

    # 投影梯度起步（取无约束解与 warm start 中的较优者）
    def cost(U):
        return 0.5 * U @ H @ U + g @ U

    U = U_free.copy()
    if cost(np.clip(U0, u_min, u_max)) < cost(np.clip(U, u_min, u_max)):
        U = U0.copy()
    U = np.clip(U, u_min, u_max)

    # 安全步长：α = 1 / λ_max(H)
    eig_max = np.linalg.eigvalsh(H)[-1]
    alpha = 1.0 / max(eig_max, 1e-6)

    for _ in range(n_iter):
        grad = H @ U + g
        U = np.clip(U - alpha * grad, u_min, u_max)
    return U


class MPCBalanceController:
    """MPC 平衡控制器（两腿独立求解）"""

    def __init__(self, config_path=CONFIG_FILE):
        from Controller import PID

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"未找到 {config_path}，请先运行 python calc_mpc_config.py 生成")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.model = MPCModelTable(
            config["L0_values"],
            config["Ad_table"], config["Bd_table"], config["P_table"],
        )
        self.leg_params = config.get("leg_params", None)

        self.Q = np.diag(config["Q"])
        self.R = np.diag(config["R"])
        self.dt = float(config["dt"])
        self.N = int(config["N"])
        self.u_min_step = np.asarray(config["u_min"], dtype=float)  # (2,)
        self.u_max_step = np.asarray(config["u_max"], dtype=float)
        self.u_min = np.tile(self.u_min_step, self.N)
        self.u_max = np.tile(self.u_max_step, self.N)

        print(f"[MPC] N={self.N}, dt={self.dt}s, "
              f"L0 ∈ [{config['L0_range']['min']:.2f}, {config['L0_range']['max']:.2f}] m")

        # 目标值
        self.L0_target = 0.2
        self.x_target = 0.0
        self.v_target = 0.0
        self.yaw_target = 0.0
        self.pitch_target = 0.0  # 机体 pitch 目标 (rad)

        # |v_target| 超过此阈值进入速度控制模式：冻结进入瞬间的位置误差，
        # 让 x_target 随 body_x 等速漂移，位置误差保持不变（不累加）
        self.v_hold_threshold = 0.05
        self._prev_v_mode = False
        self._frozen_x_err = 0.0

        # 状态估计器
        self.state = StateEstimator(self.leg_params)

        # 腿长 PID
        self.pid_L0_r = PID(p=2000.0, i=10.0, d=9000.0, integral_limit=50.0, output_limit=300.0)
        self.pid_L0_l = PID(p=2000.0, i=10.0, d=9000.0, integral_limit=50.0, output_limit=300.0)

        # yaw PID（差分轮子力矩）
        self.pid_yaw = PID(p=10.0, i=0.1, d=30.0, integral_limit=2.0, output_limit=4.0)

        # warm start（两条腿各一份）
        self._U_prev = [np.zeros(2 * self.N), np.zeros(2 * self.N)]

        # 监控
        self.T = 0.0
        self.Tp_r = 0.0
        self.Tp_l = 0.0
        self.t_count = 0

    def _solve_leg(self, leg_idx, x0):
        """对单条腿在当前 L0 下求解 MPC，返回 u_0 = [T, Tp]"""
        leg = self.state.leg[leg_idx]
        Ad, Bd, P = self.model.get(leg.L0)

        H, Fx = build_condensed(Ad, Bd, self.Q, self.R, P, self.N)
        g = Fx @ x0

        # warm start：上一步解整体左移一位，末位用 0
        U0 = self._U_prev[leg_idx]
        U_shift = np.zeros_like(U0)
        U_shift[:-2] = U0[2:]

        U = solve_qp_pgd(H, g, self.u_min, self.u_max, U_shift, n_iter=30)
        self._U_prev[leg_idx] = U
        return U[:2]  # u_0

    def compute(self, imu, motors):
        """
        参数:
            imu:    IMUData
            motors: list[MotorData] 顺序 [右前, 右后, 左前, 左后, 右轮, 左轮]
        返回:
            joint_torque[4], wheel_torque[2]
        """
        self.t_count += 1
        self.state.update(imu, motors)

        phi = self.state.body.phi
        phi_dot = self.state.body.phi_dot
        body_x = self.state.body.x
        body_vx = self.state.body.x_dot

        # 速度控制模式：冻结位置误差。进入瞬间记录 delta = body_x - x_target，
        # 之后令 x_target = body_x - delta，使误差恒为 delta（不累加）
        v_mode = abs(self.v_target) > self.v_hold_threshold
        if v_mode:
            if not self._prev_v_mode:
                self._frozen_x_err = body_x - self.x_target
            self.x_target = body_x - self._frozen_x_err
        self._prev_v_mode = v_mode

        # --- MPC → 驱动轮力矩（左右独立） ---
        wheel_torque = [0.0, 0.0]
        for i in range(2):
            leg = self.state.leg[i]
            # phi = -pitch，因此 -phi = pitch；误差 = pitch - pitch_target
            x0 = np.array([
                leg.Theta,
                leg.dTheta,
                (body_x - self.x_target),
                (body_vx - self.v_target),
                -phi - self.pitch_target,
                -phi_dot,
            ])
            
            # x0 = np.array([
            #     leg.Theta,
            #     leg.dTheta,
            #     0,
            #     0,
            #     -phi,
            #     -phi_dot,
            # ])
            
            u0 = self._solve_leg(i, x0)
            T, Tp = float(u0[0]), float(u0[1])

            wheel_torque[i] = max(-self.u_max_step[0],
                                   min(self.u_max_step[0], T))
            # wheel_torque[i] = 0
            
            if i == 0:
                self.Tp_r = -Tp
            else:
                self.Tp_l = -Tp

        # --- yaw PID 差分 ---
        yaw_correction = self.pid_yaw.calc(self.state.body.y, self.yaw_target)
        wheel_torque[0] += yaw_correction
        wheel_torque[1] -= yaw_correction

        # --- 腿长 PID + 重力前馈 ---
        ff_r = StateEstimator.gravity_feedforward(self.state.leg[0].Theta)
        F0_r = self.pid_L0_r.calc(self.state.leg[0].L0, self.L0_target) + ff_r

        ff_l = StateEstimator.gravity_feedforward(self.state.leg[1].Theta)
        F0_l = self.pid_L0_l.calc(self.state.leg[1].L0, self.L0_target) + ff_l

        # --- VMC 映射关节力矩 ---
        self.state.vmc_r.calc_torque(F0_r, self.Tp_r)
        self.state.vmc_l.calc_torque(F0_l, self.Tp_l)

        joint_torque = [
            -self.state.vmc_r.torque_set[1],  # 右前
            -self.state.vmc_r.torque_set[0],  # 右后
             self.state.vmc_l.torque_set[1],  # 左前
             self.state.vmc_l.torque_set[0],  # 左后
        ]

        return joint_torque, wheel_torque
