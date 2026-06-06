"""
响应曲线图与汇总图绘制（matplotlib）。

每个工况输出一张三联子图（PID/LQR/MPC 同图对比）：
  行1 pitch (°)；行2 工况相关跟踪量；行3 驱动轮力矩 (N·m, 含 ±4 饱和线)。
另输出一张各工况头条指标的汇总柱状图。
中文标签使用系统可用的 Noto CJK 字体。
噪声大的信号（力矩、角速度）显示时做零相位低通平滑（原始信号淡色叠底），
仅影响图表，不改 CSV 原始数据与控制行为。
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import butter, filtfilt

from . import config

# 噪声大、需要显示低通平滑的信号：驱动轮/髋部力矩、角速度状态，
# 以及机体倾角(φ/pitch)与虚拟腿摆角(θ)。
# 仅影响绘图：滤波后画为主线，原始信号以淡色叠底，保证图表诚实。
NOISY_SIGNALS = {"T_right", "T_left", "Tp_r", "Tp_l",
                 "s_theta", "s_dtheta", "s_phi", "s_dphi", "pitch",
                 "s_dx", "vx"}
# 单独指定更强滤波（更低截止频率）的信号：倾角角速度 dφ/dt
SIGNAL_CUTOFF_HZ = {"s_dphi": config.PLOT_LPF_CUTOFF_STRONG_HZ}
_DEG = 180.0 / np.pi

# 规范控制器顺序（PID/LQR/MPC），用于在图中按固定次序遍历 runs 中实际存在的控制器。
_CTRL_ORDER = list(config.CONTROLLER_LABEL)


def _present(runs):
    """返回 runs 中实际存在的控制器键，按 PID/LQR/MPC 规范顺序排列。"""
    return [ck for ck in _CTRL_ORDER if ck in runs]


def _title_controllers(runs):
    """图题里的控制器名串，如 "PID" 或 "LQR / MPC"，反映本次实际所跑控制器。"""
    return " / ".join(config.CONTROLLER_LABEL[ck] for ck in _present(runs))


def _lpf(y, fc=None):
    """零相位低通（Butterworth + filtfilt），用于显示平滑。fc 为截止频率 (Hz)。"""
    if fc is None:
        fc = config.PLOT_LPF_CUTOFF_HZ
    fs = 1.0 / config.DT_CTRL
    wn = fc / (0.5 * fs)
    b, a = butter(2, min(wn, 0.99))
    if len(y) <= 12:
        return y
    return filtfilt(b, a, y)


def set_cjk_font():
    """选择系统可用的 CJK 字体用于中文标签。"""
    prefer = ["Noto Serif CJK JP", "Noto Sans CJK JP", "Noto Serif CJK SC",
              "Source Han Serif SC", "Songti SC", "SimSun"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def _m(d):
    """绘图时间掩码：只取 t >= PLOT_START_S 的部分。"""
    return d["t"] >= config.PLOT_START_S


def _plot_sig(ax, d, key, color, lw=1.2, label=None, scale=1.0):
    """画单条信号；若属噪声大信号则零相位低通平滑为主线、原始淡色叠底。"""
    m = _m(d)
    t = d["t"][m]
    y = d[key] * scale
    if key in NOISY_SIGNALS:
        fc = SIGNAL_CUTOFF_HZ.get(key)  # None → 默认截止
        ax.plot(t, y[m], color=color, lw=0.5, alpha=0.18)            # 原始（叠底）
        ax.plot(t, _lpf(y, fc)[m], color=color, lw=lw, label=label)  # 低通平滑（主线）
    else:
        ax.plot(t, y[m], color=color, lw=lw, label=label)


# 工况 → 中间行（跟踪量）绘制方式
def _plot_track_row(ax, scenario, data, color):
    idx = scenario.index
    if idx in (1, 2):       # 位置
        _plot_sig(ax, data, "x", color)
    elif idx == 3:          # 速度
        _plot_sig(ax, data, "vx", color)
    elif idx == 4:          # 腿长
        _plot_sig(ax, data, "L0", color)
    elif idx == 5:          # 扰动工况中间行画 pitch 已在行1，这里画 Tp
        _plot_sig(ax, data, "Tp_r", color)


_TRACK_YLABEL = {
    1: "位移 x (m)", 2: "位移 x (m)", 3: "速度 vx (m/s)",
    4: "腿长 L0 (m)", 5: "髋部力矩 Tp (N·m)",
}


def plot_scenario(scenario, runs, out_dir):
    """runs: dict[ctrl_key -> data]。输出 PNG 路径。"""
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    ax_pitch, ax_track, ax_tor = axes

    for ck in _present(runs):
        data = runs[ck]
        c = config.CONTROLLER_COLOR[ck]
        lbl = config.CONTROLLER_LABEL[ck]
        _plot_sig(ax_pitch, data, "pitch", c, lw=1.2, label=lbl, scale=_DEG)
        _plot_track_row(ax_track, scenario, data, c)
        _plot_sig(ax_tor, data, "T_right", c, lw=1.0, label=lbl)

    # 参考线 / 目标线（取任一条 run 的目标时序）
    ref = next(iter(runs.values()))
    rm = _m(ref)
    t = ref["t"][rm]
    if scenario.index == 2:
        ax_track.plot(t, ref["x_target"][rm], "k--", lw=0.9, label="目标")
    elif scenario.index == 3:
        ax_track.plot(t, ref["v_target"][rm], "k--", lw=0.9, label="目标")
    elif scenario.index == 4:
        ax_track.plot(t, ref["L0_target"][rm], "k--", lw=0.9, label="目标")

    # 扰动窗口阴影 + 显示窗口裁剪（聚焦扰动前后）
    if scenario._disturbance is not None and scenario.step_time is not None:
        from .scenarios import DISTURB_DUR, DISTURB_WINDOW
        for ax in axes:
            ax.axvspan(scenario.step_time, scenario.step_time + DISTURB_DUR,
                       color="0.85", alpha=0.6, zorder=0)
            ax.set_xlim(*DISTURB_WINDOW)

    # 轮力矩饱和线
    ax_tor.axhline(config.WHEEL_TORQUE_LIMIT, color="0.5", ls=":", lw=0.8)
    ax_tor.axhline(-config.WHEEL_TORQUE_LIMIT, color="0.5", ls=":", lw=0.8)

    ax_pitch.set_ylabel("pitch (°)")
    ax_track.set_ylabel(_TRACK_YLABEL[scenario.index])
    ax_tor.set_ylabel("右轮力矩 (N·m)")
    ax_tor.set_xlabel("时间 t (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    ax_pitch.legend(loc="best", fontsize=9, ncol=3)
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）{_title_controllers(runs)} 响应对比",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    path = os.path.join(out_dir, f"case{scenario.index}_{scenario.key}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_states(scenario, runs, out_dir):
    """六维状态反馈量 x=(θ,θ̇,x,ẋ,φ,φ̇) 随时间变化（PID/LQR/MPC 同图对比）。"""
    specs = [
        ("s_theta", "θ  虚拟腿摆角 (rad)"),
        ("s_dtheta", "dθ/dt  摆角速度 (rad/s)"),
        ("s_x", "x  位移 (m)"),
        ("s_dx", "dx/dt  速度 (m/s)"),
        ("s_phi", "φ  机体倾角 (=pitch, rad)"),
        ("s_dphi", "dφ/dt  倾角速度 (rad/s)"),
    ]
    # 瞬态扰动工况：标出推力施加时间段（t=step_time 起，持续 DISTURB_DUR），
    # 并把 x 轴裁剪到扰动前后的显示窗口
    from .scenarios import DISTURB_TIME, DISTURB_DUR, DISTURB_WINDOW
    disturb_span = None
    disturb_window = None
    if scenario.index == 5:
        disturb_span = (DISTURB_TIME, DISTURB_TIME + DISTURB_DUR)
        disturb_window = DISTURB_WINDOW
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
    axes = axes.ravel()
    for ax, (key, ylabel) in zip(axes, specs):
        for ck in _present(runs):
            _plot_sig(ax, runs[ck], key, config.CONTROLLER_COLOR[ck],
                      lw=1.0, label=config.CONTROLLER_LABEL[ck])
        if disturb_span is not None:
            ax.axvspan(disturb_span[0], disturb_span[1], color="#d62728",
                       alpha=0.18, lw=0, label="扰动施加")
        if disturb_window is not None:
            ax.set_xlim(*disturb_window)
        ax.axhline(0.0, color="0.6", ls="--", lw=0.8)  # 目标值均为 0
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9, ncol=4)
    axes[4].set_xlabel("时间 t (s)")
    axes[5].set_xlabel("时间 t (s)")
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）六维状态反馈量随时间变化"
                 f"（目标值均为 0，虚线）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out_dir, f"case{scenario.index}_{scenario.key}_states.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_state_curves(scenario, runs, out_dir, specs, suffix="states"):
    """
    绘制指定状态量子集随时间变化（PID/LQR/MPC 同图对比）。
    specs: list of (key, ylabel, scale, target_key)；target_key 为 None 时画 0 参考线，
           否则取该目标时序画虚线（用于阶跃类工况）。
    """
    n = len(specs)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.0 * rows + 0.5), sharex=True)
    axes = axes.ravel()
    ref = next(iter(runs.values()))
    rm = _m(ref)
    for ax, (key, ylabel, scale, tgt) in zip(axes, specs):
        for ck in _present(runs):
            _plot_sig(ax, runs[ck], key, config.CONTROLLER_COLOR[ck],
                      lw=1.0, label=config.CONTROLLER_LABEL[ck], scale=scale)
        if tgt is not None:
            ax.plot(ref["t"][rm], ref[tgt][rm] * scale, "k--", lw=0.9, label="目标")
        else:
            ax.axhline(0.0, color="0.6", ls="--", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:        # 隐藏多余子图
        ax.set_visible(False)
    axes[0].legend(loc="best", fontsize=9, ncol=3)
    for ax in axes[max(0, n - 2):n]:
        ax.set_xlabel("时间 t (s)")
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）状态量随时间变化", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out_dir, f"case{scenario.index}_{scenario.key}_{suffix}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_states_legtrack(scenario, runs, out_dir):
    """六维状态反馈量 + 腿长 L0 跟踪 + 跟踪误差，合并为一张 8 面板图（用于腿长动态工况）。"""
    state_specs = [
        ("s_theta", "θ  虚拟腿摆角 (rad)"),
        ("s_dtheta", "dθ/dt  摆角速度 (rad/s)"),
        ("s_x", "x  位移 (m)"),
        ("s_dx", "dx/dt  速度 (m/s)"),
        ("s_phi", "φ  机体倾角 (=pitch, rad)"),
        ("s_dphi", "dφ/dt  倾角速度 (rad/s)"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    axes = axes.ravel()
    ref = next(iter(runs.values()))
    rm = _m(ref)

    # 前 6 格：六维状态（目标均为 0）
    for ax, (key, ylabel) in zip(axes[:6], state_specs):
        for ck in _present(runs):
            _plot_sig(ax, runs[ck], key, config.CONTROLLER_COLOR[ck],
                      lw=1.0, label=config.CONTROLLER_LABEL[ck])
        ax.axhline(0.0, color="0.6", ls="--", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # 第 7 格：腿长 L0 实际 vs 目标
    ax_l0 = axes[6]
    for ck in _present(runs):
        d = runs[ck]; m = _m(d)
        ax_l0.plot(d["t"][m], d["L0"][m], color=config.CONTROLLER_COLOR[ck],
                   lw=1.3, label=config.CONTROLLER_LABEL[ck])
    ax_l0.plot(ref["t"][rm], ref["L0_target"][rm], "k--", lw=1.1, label="目标 L0")
    ax_l0.set_ylabel("腿长 L0 (m)")
    ax_l0.grid(True, alpha=0.3)

    # 第 8 格：腿长跟踪误差
    ax_err = axes[7]
    for ck in _present(runs):
        d = runs[ck]; m = _m(d)
        ax_err.plot(d["t"][m], (d["L0"] - d["L0_target"])[m],
                    color=config.CONTROLLER_COLOR[ck], lw=1.1)
    ax_err.axhline(0.0, color="0.6", ls="--", lw=0.8)
    ax_err.set_ylabel("L0 跟踪误差 (m)")
    ax_err.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=9, ncol=3)
    axes[6].set_xlabel("时间 t (s)")
    axes[7].set_xlabel("时间 t (s)")
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）六维状态反馈量 + 腿长跟踪"
                 f"（状态目标均为 0）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out_dir, f"case{scenario.index}_{scenario.key}_states.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_torque_compare(scenario, runs, out_dir, key, ylabel, suffix,
                        saturation=False):
    """
    单信号力矩对比图：3 种控制器输出的同一路力矩随时间变化叠在一张图上。
    key: "Tp_r"（右髋部力矩）或 "T_right"（右驱动轮力矩）。
    saturation=True 时叠加 ±WHEEL_TORQUE_LIMIT 饱和线（仅轮力矩需要）。
    噪声大的力矩信号沿用 _plot_sig 的零相位低通平滑（原始淡色叠底）。
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for ck in _present(runs):
        _plot_sig(ax, runs[ck], key, config.CONTROLLER_COLOR[ck],
                  lw=1.2, label=config.CONTROLLER_LABEL[ck])

    # 扰动工况：标出扰动窗口并裁剪显示区间，与其它图保持一致
    if scenario._disturbance is not None and scenario.step_time is not None:
        from .scenarios import DISTURB_DUR, DISTURB_WINDOW
        ax.axvspan(scenario.step_time, scenario.step_time + DISTURB_DUR,
                   color="0.85", alpha=0.6, zorder=0)
        ax.set_xlim(*DISTURB_WINDOW)

    if saturation:
        ax.axhline(config.WHEEL_TORQUE_LIMIT, color="0.5", ls=":", lw=0.8)
        ax.axhline(-config.WHEEL_TORQUE_LIMIT, color="0.5", ls=":", lw=0.8)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("时间 t (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=3)
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）{_title_controllers(runs)} "
                 f"{ylabel}对比", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    path = os.path.join(out_dir, f"case{scenario.index}_{scenario.key}_{suffix}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_summary(scenarios, metrics_table, out_dir):
    """各工况头条指标的分组柱状图。metrics_table[ck][scenario.index] -> headline。"""
    labels = [f"{s.index}\n{s.title}" for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, ck in enumerate(config.CONTROLLERS):
        vals = [metrics_table[ck][s.index].get("headline", np.nan) for s in scenarios]
        ax.bar(x + (i - 1) * width, vals, width,
               label=config.CONTROLLER_LABEL[ck],
               color=config.CONTROLLER_COLOR[ck])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("头条指标（各工况量纲见表）")
    ax.set_title("各工况头条指标对比（LQR / MPC）")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_headline.png")
    fig.savefig(path)
    plt.close(fig)
    return path
