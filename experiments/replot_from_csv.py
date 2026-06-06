"""
从 output/data/ 的已存 CSV 重绘每工况 8 图组合，不重跑 MuJoCo 仿真。

用途：仅当绘图代码（标题/样式）变更、而仿真数据未变时，用本脚本以原始 CSV
快速重生成 output/figures/caseN_*_states.png，避免重跑耗时仿真。数据与 metrics.json 保持一致。

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
        # 每工况一张 8 图组合：6 状态反馈量 + 2 路控制输出(Tp/T)
        sp = plotting.plot_states(s, runs, config.FIG_DIR)
        print(f"[重绘] 工况{s.index} {s.title} → {os.path.relpath(sp, config.REPO_ROOT)}")
        # 跟踪类工况（2 位置 / 3 速度 / 4 腿长）同步重绘跟踪图
        if s.index in (2, 3, 4):
            tp = plotting.plot_tracking(s, runs, config.FIG_DIR)
            print(f"        跟踪图 → {os.path.relpath(tp, config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
