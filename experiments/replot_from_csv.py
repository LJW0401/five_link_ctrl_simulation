"""
从 output/data/ 的已存 CSV 重绘工况响应对比图，不重跑 MuJoCo 仿真。

用途：仅当绘图代码（标题/样式）变更、而仿真数据未变时，用本脚本以原始 CSV
快速重生成 output/figures/caseN_*.png，避免重跑耗时仿真。数据与 metrics.json 保持一致。

依赖：experiments.config（路径与控制器清单）、experiments.scenarios（工况定义）、
      experiments.plotting（绘图）。
运行：python -m experiments.replot_from_csv
"""

import os

import numpy as np

from . import config
from . import plotting
from .scenarios import get_scenarios


def _load_csv(path):
    """读 CSV 为 {列名: ndarray} 字典，键与仿真时内存 data 字典一致。"""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return {name: arr[:, i] for i, name in enumerate(header)}


def main():
    plotting.set_cjk_font()
    for s in get_scenarios():
        runs = {}
        for ck in config.CONTROLLERS:
            csv_path = os.path.join(config.DATA_DIR, f"case{s.index}_{s.key}_{ck}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"缺少 CSV：{csv_path}")
            runs[ck] = _load_csv(csv_path)
        fig_path = plotting.plot_scenario(s, runs, config.FIG_DIR)
        print(f"[重绘] 工况{s.index} {s.title} → {os.path.relpath(fig_path, config.REPO_ROOT)}")
        # 同步重绘两张力矩对比图（右髋部力矩 Tp、右驱动轮力矩 T）
        plotting.plot_torque_compare(
            s, runs, config.FIG_DIR, "Tp_r", "右髋部力矩 Tp (N·m)", "Tp")
        plotting.plot_torque_compare(
            s, runs, config.FIG_DIR, "T_right", "右驱动轮力矩 T (N·m)", "T",
            saturation=True)


if __name__ == "__main__":
    main()
