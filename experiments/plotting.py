"""
响应曲线图与汇总图绘制（matplotlib）。

每个工况统一输出一张 8 图组合（4×2，PID/LQR/MPC 同图对比）：
  6 个状态反馈量 (θ,θ̇,x,ẋ,φ,φ̇) + 2 路控制输出（右髋部力矩 Tp、右驱动轮力矩 T，
  后者含 ±4 N·m 饱和线）。各状态目标统一为 0（虚线）。
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


def _torque_ax(ax, runs, key, ylabel, saturation=False,
               disturb_span=None, disturb_window=None):
    """在给定子图上画 3 种控制器输出的同一路力矩（用于六状态图末尾追加的力矩面板）。
    key: "Tp_r"（右髋部力矩）或 "T_right"（右驱动轮力矩）；
    saturation=True 叠加 ±WHEEL_TORQUE_LIMIT 饱和线（仅轮力矩需要）；
    扰动工况传入 disturb_span/disturb_window 以与状态面板一致地标注与裁剪。
    力矩为噪声大信号，_plot_sig 自动做零相位低通平滑（原始淡色叠底）。"""
    for ck in _present(runs):
        _plot_sig(ax, runs[ck], key, config.CONTROLLER_COLOR[ck],
                  lw=1.0, label=config.CONTROLLER_LABEL[ck])
    if disturb_span is not None:
        ax.axvspan(disturb_span[0], disturb_span[1], color="#d62728",
                   alpha=0.18, lw=0)
    if disturb_window is not None:
        ax.set_xlim(*disturb_window)
    if saturation:
        ax.axhline(config.WHEEL_TORQUE_LIMIT, color="0.5", ls=":", lw=0.8)
        ax.axhline(-config.WHEEL_TORQUE_LIMIT, color="0.5", ls=":", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_states(scenario, runs, out_dir):
    """每工况统一输出的 8 图组合（4×2）：六维状态反馈量 x=(θ,θ̇,x,ẋ,φ,φ̇) +
    两路控制输出（右髋部力矩 Tp、右驱动轮力矩 T），PID/LQR/MPC 同图对比。

    六维状态均为「相对目标的偏差」（harness 中 s_x=x−x_target、s_dx=ẋ−v_target，
    其余目标本为 0），故各状态面板的目标线统一为 0（虚线）；位置/速度跟踪工况的
    跟踪表现即体现为对应偏差收敛到 0。瞬态扰动工况叠加扰动窗口阴影并裁剪显示区间。"""
    specs = [
        ("s_theta", "θ  虚拟腿摆角 (rad)"),
        ("s_dtheta", "dθ/dt  摆角速度 (rad/s)"),
        ("s_x", "x  位移偏差 (m)"),
        ("s_dx", "dx/dt  速度偏差 (m/s)"),
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
    # 4×2：前 6 格六维状态，末 2 格追加右髋部力矩 Tp 与右驱动轮力矩 T
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    axes = axes.ravel()
    for ax, (key, ylabel) in zip(axes[:6], specs):
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
    _torque_ax(axes[6], runs, "Tp_r", "Tp  右髋部力矩 (N·m)",
               disturb_span=disturb_span, disturb_window=disturb_window)
    _torque_ax(axes[7], runs, "T_right", "T  右驱动轮力矩 (N·m)", saturation=True,
               disturb_span=disturb_span, disturb_window=disturb_window)
    axes[0].legend(loc="best", fontsize=9, ncol=4)
    axes[6].set_xlabel("时间 t (s)")
    axes[7].set_xlabel("时间 t (s)")
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）六维状态反馈量与控制力矩"
                 f"随时间变化（状态目标均为 0，虚线）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out_dir, f"case{scenario.index}_{scenario.key}_states.png")
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
