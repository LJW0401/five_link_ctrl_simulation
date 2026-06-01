"""
从单次仿真时序计算论文表 4 所需的量化指标。

约定：
  - pitch/yaw 时序单位为 rad，指标按需转 deg
  - 稳态类指标在 t >= scenario.settle 的窗口上统计
  - 瞬态类指标围绕 scenario.step_time 统计
所有指标均基于 MuJoCo 真实状态（无 IMU 噪声）。
"""

import numpy as np

from . import config


def _window(t, lo=None, hi=None):
    m = np.ones_like(t, dtype=bool)
    if lo is not None:
        m &= (t >= lo)
    if hi is not None:
        m &= (t < hi)
    return m


def _sat_frac(data, mask):
    """轮力矩饱和占比 (%)：|T| 触及上限的控制步比例。"""
    lim = config.SATURATION_FRAC * config.WHEEL_TORQUE_LIMIT
    peak = np.maximum(np.abs(data["T_right"]), np.abs(data["T_left"]))
    sel = peak[mask]
    if sel.size == 0:
        return float("nan")
    return 100.0 * np.mean(sel >= lim)


def compute_metrics(data, scenario):
    """返回该工况的指标字典（含通用指标 + 头条指标）。"""
    t = data["t"]
    pitch_deg = np.degrees(data["pitch"])
    steady = _window(t, lo=scenario.settle)

    m = {}
    # ---- 通用指标 ----
    m["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch_deg[steady] ** 2))) if steady.any() else float("nan")
    m["pitch_peak_deg"] = float(np.max(np.abs(pitch_deg)))
    # 倒伏判定：稳态窗口内出现 > 45° 的姿态即视为失稳倒伏（用于诚实标注）
    m["fell"] = bool(np.max(np.abs(pitch_deg[steady])) > 45.0) if steady.any() else False
    m["max_wheel_torque"] = float(np.max(np.maximum(np.abs(data["T_right"]), np.abs(data["T_left"]))))
    m["sat_frac_pct"] = _sat_frac(data, _window(t, lo=scenario.settle))
    m["solve_ms_mean"] = float(np.mean(data["solve_ms"]))

    # ---- 工况头条指标 ----
    idx = scenario.index
    if idx == 1:  # 平衡保持：稳态 pitch RMS
        m["headline"] = m["pitch_rms_deg"]

    elif idx == 2:  # 位置阶跃：上升时间 + pitch 反向超调
        st = scenario.step_time
        target = 0.5
        after = _window(t, lo=st)
        x_after = data["x"][after]
        t_after = t[after]
        reach = np.where(x_after >= 0.9 * target)[0]
        m["rise_time_s"] = float(t_after[reach[0]] - st) if reach.size else float("nan")
        # 反向超调：阶跃后 pitch 与前进方向相反的最大幅值
        tr = _window(t, lo=st, hi=st + 3.0)
        seg = pitch_deg[tr]
        m["pitch_reverse_overshoot_deg"] = float(np.max(np.abs(seg))) if seg.size else float("nan")
        m["headline"] = m["rise_time_s"]

    elif idx == 3:  # 速度跟踪：稳态速度误差
        st = scenario.step_time
        v_des = 0.3
        ss = _window(t, lo=st + 3.0)
        verr = np.abs(data["vx"][ss] - v_des)
        m["vel_ss_err"] = float(np.mean(verr)) if verr.size else float("nan")
        m["headline"] = m["vel_ss_err"]

    elif idx == 4:  # yaw 阶跃：转向期间 pitch 偏差
        st = scenario.step_time
        turn = _window(t, lo=st, hi=st + 3.0)
        seg = np.abs(pitch_deg[turn])
        m["pitch_dev_turn_deg"] = float(np.max(seg)) if seg.size else float("nan")
        # yaw 稳态误差（deg）
        yaw_des = np.degrees(data["yaw_target"][-1])
        yss = _window(t, lo=scenario.duration - 2.0)
        m["yaw_ss_err_deg"] = float(np.mean(np.abs(np.degrees(data["yaw"][yss]) - yaw_des)))
        m["headline"] = m["pitch_dev_turn_deg"]

    elif idx == 5:  # 腿长正弦：pitch 振幅
        osc = _window(t, lo=scenario.settle)
        seg = pitch_deg[osc]
        if seg.size:
            amp = 0.5 * (np.percentile(seg, 97.5) - np.percentile(seg, 2.5))
        else:
            amp = float("nan")
        m["pitch_amp_deg"] = float(amp)
        # 腿长跟踪 RMS 误差
        l0err = data["L0"][osc] - data["L0_target"][osc]
        m["L0_track_rms_m"] = float(np.sqrt(np.mean(l0err ** 2))) if l0err.size else float("nan")
        # 头条用 pitch RMS（相对竖直的偏离），诚实反映「卡死大角度倾斜」也算失败，
        # 而非用振幅 —— 振幅会把恒定大倾角误判为「波动小=表现好」。
        m["headline"] = m["pitch_rms_deg"]

    elif idx == 6:  # 瞬态扰动：pitch 峰值 + 饱和占比
        st = scenario.step_time
        after = _window(t, lo=st)
        seg = np.abs(pitch_deg[after])
        m["pitch_peak_after_deg"] = float(np.max(seg)) if seg.size else float("nan")
        rec = _window(t, lo=st, hi=st + 2.0)
        m["sat_frac_recovery_pct"] = _sat_frac(data, rec)
        m["headline"] = m["pitch_peak_after_deg"]

    return m
