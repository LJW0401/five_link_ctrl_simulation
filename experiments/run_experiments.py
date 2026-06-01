"""
实验编排入口：跑全部 (工况 × 控制器)，导出数据、图表与汇总表到 output/。

用法（从仓库根目录）：
    python -m experiments.run_experiments              # 跑全部
    python -m experiments.run_experiments --scenarios 1 6   # 只跑指定工况
    python -m experiments.run_experiments --controllers lqr mpc

输出：
    output/data/    每个 (工况,控制器) 一份原始时序 CSV
    output/figures/ 每个工况一张三联响应对比图 + 头条指标汇总图
    output/tables/  汇总指标 summary.csv / summary.md / summary.tex
    output/metrics.json  全部指标的机器可读副本
"""

import os
import sys
import json
import argparse

import numpy as np

from . import config
from .scenarios import get_scenarios
from .harness import run_one
from .metrics import compute_metrics
from . import plotting


# 汇总表行：(工况名, 指标名+单位, 工况编号, metrics 键, 格式)
SUMMARY_ROWS = [
    ("1 平衡保持", "pitch RMS / °", 1, "pitch_rms_deg", "{:.2f}"),
    ("2 位置阶跃", "上升时间 / s", 2, "rise_time_s", "{:.2f}"),
    ("2 位置阶跃", "pitch 反向超调 / °", 2, "pitch_reverse_overshoot_deg", "{:.1f}"),
    ("3 速度跟踪", "稳态误差 / (m·s⁻¹)", 3, "vel_ss_err", "{:.3f}"),
    ("4 yaw 阶跃", "转向期间 pitch 偏差 / °", 4, "pitch_dev_turn_deg", "{:.2f}"),
    ("5 L0 正弦", "pitch RMS / °", 5, "pitch_rms_deg", "{:.2f}"),
    ("6 瞬态扰动", "pitch 峰值 / °", 6, "pitch_peak_after_deg", "{:.1f}"),
    ("6 瞬态扰动", "轮力矩饱和占比 / %", 6, "sat_frac_recovery_pct", "{:.0f}"),
]

CSV_COLUMNS = [
    "t", "pitch", "pitch_cmd", "yaw", "x", "x_true", "vx",
    "L0", "L0_target", "x_target", "v_target", "yaw_target",
    "T_right", "T_left", "Tp_r", "Tp_l", "solve_ms",
]


def _ensure_dirs():
    for d in (config.OUTPUT_DIR, config.DATA_DIR, config.FIG_DIR, config.TABLE_DIR):
        os.makedirs(d, exist_ok=True)


def _save_csv(data, scenario, ck):
    cols = [data[c] for c in CSV_COLUMNS]
    arr = np.column_stack(cols)
    path = os.path.join(config.DATA_DIR, f"case{scenario.index}_{scenario.key}_{ck}.csv")
    header = ",".join(CSV_COLUMNS)
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.6g")
    return path


def _fmt(val, fmt):
    if val is None or (isinstance(val, float) and (np.isnan(val))):
        return "—"
    return fmt.format(val)


def _write_summary_tables(scenarios, metrics_table):
    # 求解耗时行（各工况平均）
    solve_row = {}
    for ck in config.CONTROLLERS:
        vals = [metrics_table[ck][s.index]["solve_ms_mean"] for s in scenarios]
        solve_row[ck] = float(np.mean(vals))

    # ---- CSV ----
    csv_path = os.path.join(config.TABLE_DIR, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("工况,指标," + ",".join(config.CONTROLLER_LABEL[c] for c in config.CONTROLLERS) + "\n")
        for case, metric, idx, key, fmt in SUMMARY_ROWS:
            vals = [_fmt(metrics_table[c][idx].get(key), fmt) for c in config.CONTROLLERS]
            f.write(f"{case},{metric}," + ",".join(vals) + "\n")
        f.write("平均单步求解耗时,/ ms," +
                ",".join("{:.3f}".format(solve_row[c]) for c in config.CONTROLLERS) + "\n")

    # ---- Markdown ----
    md_path = os.path.join(config.TABLE_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 三类控制器六工况仿真指标汇总\n\n")
        f.write("| 工况 | 指标 | " + " | ".join(config.CONTROLLER_LABEL[c] for c in config.CONTROLLERS) + " |\n")
        f.write("|---|---|" + "---|" * len(config.CONTROLLERS) + "\n")
        for case, metric, idx, key, fmt in SUMMARY_ROWS:
            vals = [_fmt(metrics_table[c][idx].get(key), fmt) for c in config.CONTROLLERS]
            f.write(f"| {case} | {metric} | " + " | ".join(vals) + " |\n")
        f.write("| 平均单步求解耗时 | / ms | " +
                " | ".join("{:.3f}".format(solve_row[c]) for c in config.CONTROLLERS) + " |\n")
        f.write("\n> 数据由 `python -m experiments.run_experiments` 在 MuJoCo MJCF 模型上实测生成。"
                "控制器接收叠加白噪声（σ={:.2f}）的 IMU；下表指标统计于 **无噪声真实状态**。\n".format(config.IMU_NOISE_STD))

        # ---- 稳定性矩阵（峰值 pitch / 是否倒伏）----
        from .scenarios import get_scenarios as _gs
        f.write("\n## 稳定性矩阵（稳态窗口内 |pitch| 峰值，✗ 表示 >45° 倒伏）\n\n")
        f.write("| 工况 | " + " | ".join(config.CONTROLLER_LABEL[c] for c in config.CONTROLLERS) + " |\n")
        f.write("|---|" + "---|" * len(config.CONTROLLERS) + "\n")
        for s in _gs():
            cells = []
            for c in config.CONTROLLERS:
                mm = metrics_table[c][s.index]
                tag = "✗倒伏 " if mm.get("fell") else ""
                cells.append(f"{tag}{mm['pitch_peak_deg']:.1f}°")
            f.write(f"| {s.index} {s.title} | " + " | ".join(cells) + " |\n")

    # ---- LaTeX（booktabs，可直接替换论文表 4 数据）----
    tex_path = os.path.join(config.TABLE_DIR, "summary.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% 由 experiments/run_experiments.py 自动生成，可替换 main.tex 中 tab:compare 数据\n")
        f.write("\\begin{tabular}{l l c c c}\n\\toprule\n")
        f.write("\\textbf{工况} & \\textbf{指标} & \\textbf{PID} & \\textbf{LQR} & \\textbf{MPC} \\\\\n")
        f.write("\\midrule\n")
        for case, metric, idx, key, fmt in SUMMARY_ROWS:
            vals = [_fmt(metrics_table[c][idx].get(key), fmt) for c in config.CONTROLLERS]
            f.write(f"{case} & {metric} & " + " & ".join(vals) + " \\\\\n")
        f.write("平均单步求解耗时 & / ms & " +
                " & ".join("{:.3f}".format(solve_row[c]) for c in config.CONTROLLERS) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    return csv_path, md_path, tex_path


def _write_readme(scenarios):
    path = os.path.join(config.OUTPUT_DIR, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 实验输出 output/\n\n")
        f.write("由 `python -m experiments.run_experiments` 在 MuJoCo（MJCF/env.xml）上实测生成。\n\n")
        f.write("## 目录\n\n")
        f.write("- `data/` 原始时序 CSV，命名 `caseN_<工况>_<控制器>.csv`，列见首行表头\n")
        f.write("- `figures/` 每个工况一张三类控制器响应对比图 + `summary_headline.png`\n")
        f.write("- `tables/` 汇总指标 `summary.{csv,md,tex}`（tex 可直接替换论文表 4 数据）\n")
        f.write("- `metrics.json` 全部指标的机器可读副本\n\n")
        f.write("## 工况清单\n\n")
        for s in scenarios:
            f.write(f"- 工况 {s.index} {s.title}：时长 {s.duration:.0f}s，初始 L0={s.init_L0:.2f}m\n")
        f.write("\n## 复现\n\n```bash\npython -m experiments.run_experiments\n```\n")
        f.write(f"\nIMU 噪声 σ={config.IMU_NOISE_STD}（方差 {config.IMU_NOISE_STD**2:.0e}），"
                f"固定种子 {config.NOISE_SEED}；控制周期 {config.DT_CTRL*1e3:.0f} ms；"
                f"每次工况前有 {config.WARMUP_S:.0f}s 预平衡热身（不记录），热身末清零里程计。\n")

        f.write("\n## 实测结论与注意事项（如实记录，与论文表 4 乐观预期值有差异）\n\n")
        f.write("- **平衡/姿态类工况**：LQR 最优（pitch RMS ≈1.5°），MPC 次之（≈5°），"
                "PID 明显最差（稳态有大 pitch 偏置）。这一相对优劣与论文论点一致，"
                "但绝对量级远大于表 4 的乐观估计值。\n")
        f.write("- **位置/速度跟踪较弱**：三类控制器都能保持平衡，但位置外环很弱——"
                "即使目标置零，整车也会缓慢溜车（LQR 收敛到约 −1.5m 的稳态偏移、PID 持续漂移）。"
                "故位置阶跃工况普遍无法到达 +0.5m（上升时间记为「—」），速度稳态误差也偏大。\n")
        f.write("- **腿长动态工况（工况 5）最苛刻**：仅 PID 跟住了 L0 正弦（±22° pitch 摆动）；"
                "LQR 卡死在约 −27.5° 倾角、MPC 直接倒伏。详见稳定性矩阵。\n")
        f.write("- 指标基于 MuJoCo 无噪声真实状态统计；倒伏（稳态 |pitch|>45°）已在 summary.md 标注。\n")
        f.write("- 状态估计 `dTheta` 改用陀螺仪角速度（而非对含噪 pitch 角差分），"
                "无噪声时与原实现数值等价，但避免了噪声经微分被放大；这是采集层修正，未改控制律。\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="五连杆轮腿机器人三算法对比实验")
    parser.add_argument("--scenarios", nargs="*", type=int, default=None,
                        help="只跑指定工况编号（默认全部）")
    parser.add_argument("--controllers", nargs="*", default=None,
                        help="只跑指定控制器 pid/lqr/mpc（默认全部）")
    args = parser.parse_args(argv)

    scenarios = [s for s in get_scenarios()
                 if args.scenarios is None or s.index in args.scenarios]
    controllers = args.controllers or config.CONTROLLERS

    _ensure_dirs()
    plotting.set_cjk_font()

    metrics_table = {ck: {} for ck in config.CONTROLLERS}
    metrics_json = {}

    for s in scenarios:
        runs = {}
        for ck in controllers:
            print(f"[运行] 工况{s.index} {s.title} × {config.CONTROLLER_LABEL[ck]} ...",
                  flush=True)
            data = run_one(ck, s)
            runs[ck] = data
            _save_csv(data, s, ck)
            mt = compute_metrics(data, s)
            metrics_table[ck][s.index] = mt
            metrics_json[f"case{s.index}_{ck}"] = mt
        fig_path = plotting.plot_scenario(s, runs, config.FIG_DIR)
        print(f"       → 图表 {os.path.relpath(fig_path, config.REPO_ROOT)}")

    # 汇总（仅当三控制器全跑时才出汇总图/表）
    if set(controllers) == set(config.CONTROLLERS) and len(scenarios) == len(get_scenarios()):
        plotting.plot_summary(scenarios, metrics_table, config.FIG_DIR)
        csv_p, md_p, tex_p = _write_summary_tables(scenarios, metrics_table)
        _write_readme(scenarios)
        print(f"[汇总] {os.path.relpath(csv_p, config.REPO_ROOT)} / .md / .tex")

    with open(os.path.join(config.OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    print("[完成] 输出目录:", os.path.relpath(config.OUTPUT_DIR, config.REPO_ROOT))


if __name__ == "__main__":
    main()
