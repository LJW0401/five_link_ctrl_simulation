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

# 噪声大、需要显示低通平滑的信号（驱动轮/髋部力矩、角速度状态）。
# 仅影响绘图：滤波后画为主线，原始信号以淡色叠底，保证图表诚实。
NOISY_SIGNALS = {"T_right", "T_left", "Tp_r", "Tp_l", "s_dtheta", "s_dphi"}
_DEG = 180.0 / np.pi


def _lpf(y):
    """零相位低通（Butterworth + filtfilt），用于显示平滑。"""
    fs = 1.0 / config.DT_CTRL
    wn = config.PLOT_LPF_CUTOFF_HZ / (0.5 * fs)
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
        ax.plot(t, y[m], color=color, lw=0.5, alpha=0.18)        # 原始（叠底）
        ax.plot(t, _lpf(y)[m], color=color, lw=lw, label=label)  # 低通平滑（主线）
    else:
        ax.plot(t, y[m], color=color, lw=lw, label=label)


# 工况 → 中间行（跟踪量）绘制方式
def _plot_track_row(ax, scenario, data, color):
    idx = scenario.index
    if idx in (1, 2):       # 位置
        _plot_sig(ax, data, "x", color)
    elif idx == 3:          # 速度
        _plot_sig(ax, data, "vx", color)
    elif idx == 4:          # yaw
        _plot_sig(ax, data, "yaw", color, scale=_DEG)
    elif idx == 5:          # 腿长
        _plot_sig(ax, data, "L0", color)
    elif idx == 6:          # 扰动工况中间行画 pitch 已在行1，这里画 Tp
        _plot_sig(ax, data, "Tp_r", color)


_TRACK_YLABEL = {
    1: "位移 x (m)", 2: "位移 x (m)", 3: "速度 vx (m/s)",
    4: "yaw (°)", 5: "腿长 L0 (m)", 6: "髋部力矩 Tp (N·m)",
}


def plot_scenario(scenario, runs, out_dir):
    """runs: dict[ctrl_key -> data]。输出 PNG 路径。"""
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    ax_pitch, ax_track, ax_tor = axes

    for ck in config.CONTROLLERS:
        if ck not in runs:
            continue
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
        ax_track.plot(t, np.degrees(ref["yaw_target"][rm]), "k--", lw=0.9, label="目标")
    elif scenario.index == 5:
        ax_track.plot(t, ref["L0_target"][rm], "k--", lw=0.9, label="目标")

    # 扰动窗口阴影
    if scenario._disturbance is not None and scenario.step_time is not None:
        for ax in axes:
            ax.axvspan(scenario.step_time, scenario.step_time + 0.1,
                       color="0.85", alpha=0.6, zorder=0)

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
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）三类控制器响应对比",
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
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
    axes = axes.ravel()
    for ax, (key, ylabel) in zip(axes, specs):
        for ck in config.CONTROLLERS:
            if ck not in runs:
                continue
            _plot_sig(ax, runs[ck], key, config.CONTROLLER_COLOR[ck],
                      lw=1.0, label=config.CONTROLLER_LABEL[ck])
        ax.axhline(0.0, color="0.6", ls="--", lw=0.8)  # 目标值均为 0
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9, ncol=3)
    axes[4].set_xlabel("时间 t (s)")
    axes[5].set_xlabel("时间 t (s)")
    fig.suptitle(f"工况 {scenario.index}（{scenario.title}）六维状态反馈量随时间变化"
                 f"（目标值均为 0，虚线）", fontsize=13)
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
    ax.set_title("六类工况头条指标对比（PID / LQR / MPC）")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_headline.png")
    fig.savefig(path)
    plt.close(fig)
    return path
