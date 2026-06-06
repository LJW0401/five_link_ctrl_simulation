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
    """每工况输出的状态+输出组合图（PID/LQR/MPC 同图对比）：
      前 6 格为六维状态反馈量 x=(θ,θ̇,x,ẋ,φ,φ̇)，末 2 格为右髋部力矩 Tp 与
      右驱动轮力矩 T（T 含 ±4 N·m 饱和线）。一般工况 4×2=8 图。

    跟踪类工况把绝对量与目标叠在对应变量面板里（不再单出跟踪图）：
      工况 2 位置阶跃 → x 面板画绝对位移 x 与目标阶跃虚线；
      工况 3 速度跟踪 → ẋ 面板画绝对速度 vx 与目标虚线。
    其余状态面板目标恒为 0（harness 中 s_x=x−x_target、s_dx=ẋ−v_target），画 0 虚线。
    工况 4 腿长动态 → 在六状态后追加「腿长 L0 跟踪 + 跟踪误差」两格，扩为 5×2=10 图。
    瞬态扰动工况叠加扰动窗口阴影并裁剪显示区间。返回输出路径。"""
    idx = scenario.index
    # 六维状态面板规格：(信号键, 轴标签, 目标键)；目标键 None → 画 0 参考线，
    # 否则画该目标时序虚线（位置/速度跟踪工况把绝对量与目标叠进对应变量面板）
    specs = [
        ("s_theta", "θ  虚拟腿摆角 (rad)", None),
        ("s_dtheta", "dθ/dt  摆角速度 (rad/s)", None),
        ("s_x", "x  位移偏差 (m)", None),
        ("s_dx", "dx/dt  速度偏差 (m/s)", None),
        ("s_phi", "φ  机体倾角 (=pitch, rad)", None),
        ("s_dphi", "dφ/dt  倾角速度 (rad/s)", None),
    ]
    if idx == 2:
        specs[2] = ("x", "x  位移 (m)", "x_target")
    elif idx == 3:
        specs[3] = ("vx", "dx/dt  速度 (m/s)", "v_target")

    # 瞬态扰动工况：标出推力施加时间段并把 x 轴裁剪到扰动前后的显示窗口
    from .scenarios import DISTURB_TIME, DISTURB_DUR, DISTURB_WINDOW
    disturb_span = None
    disturb_window = None
    if idx == 5:
        disturb_span = (DISTURB_TIME, DISTURB_TIME + DISTURB_DUR)
        disturb_window = DISTURB_WINDOW

    legtrack = (idx == 4)          # 腿长动态工况追加 L0 跟踪+误差两格 → 10 图
    rows = 5 if legtrack else 4
    fig, axes = plt.subplots(rows, 2, figsize=(11, 13.5 if legtrack else 11), sharex=True)
    axes = axes.ravel()
    ref = next(iter(runs.values()))
    rm = _m(ref)

    for ax, (key, ylabel, tgt) in zip(axes[:6], specs):
        for ck in _present(runs):
            _plot_sig(ax, runs[ck], key, config.CONTROLLER_COLOR[ck],
                      lw=1.0, label=config.CONTROLLER_LABEL[ck])
        if disturb_span is not None:
            ax.axvspan(disturb_span[0], disturb_span[1], color="#d62728",
                       alpha=0.18, lw=0, label="扰动施加")
        if disturb_window is not None:
            ax.set_xlim(*disturb_window)
        if tgt is not None:
            ax.plot(ref["t"][rm], ref[tgt][rm], "k--", lw=0.9, label="目标")
        else:
            ax.axhline(0.0, color="0.6", ls="--", lw=0.8)  # 目标值为 0
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # 工况 4：第 7、8 格画腿长 L0 跟踪与跟踪误差
    if legtrack:
        ax_l0 = axes[6]
        for ck in _present(runs):
            d = runs[ck]; m = _m(d)
            ax_l0.plot(d["t"][m], d["L0"][m], color=config.CONTROLLER_COLOR[ck],
                       lw=1.2, label=config.CONTROLLER_LABEL[ck])
        ax_l0.plot(ref["t"][rm], ref["L0_target"][rm], "k--", lw=1.0, label="目标 L0")
        ax_l0.set_ylabel("腿长 L0 (m)")
        ax_l0.grid(True, alpha=0.3)
        ax_err = axes[7]
        for ck in _present(runs):
            d = runs[ck]; m = _m(d)
            ax_err.plot(d["t"][m], (d["L0"] - d["L0_target"])[m],
                        color=config.CONTROLLER_COLOR[ck], lw=1.0)
        ax_err.axhline(0.0, color="0.6", ls="--", lw=0.8)
        ax_err.set_ylabel("L0 跟踪误差 (m)")
        ax_err.grid(True, alpha=0.3)
        tp_i, t_i = 8, 9
    else:
        tp_i, t_i = 6, 7

    _torque_ax(axes[tp_i], runs, "Tp_r", "Tp  右髋部力矩 (N·m)",
               disturb_span=disturb_span, disturb_window=disturb_window)
    _torque_ax(axes[t_i], runs, "T_right", "T  右驱动轮力矩 (N·m)", saturation=True,
               disturb_span=disturb_span, disturb_window=disturb_window)

    axes[0].legend(loc="best", fontsize=9, ncol=4)
    axes[tp_i].set_xlabel("时间 t (s)")
    axes[t_i].set_xlabel("时间 t (s)")
    title_extra = " + 腿长跟踪" if legtrack else ""
    fig.suptitle(f"工况 {idx}（{scenario.title}）六维状态反馈量{title_extra}与控制力矩"
                 f"随时间变化", fontsize=13)
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
