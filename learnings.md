# Learnings

## 2026-06-06

### 快速功能：监控记录后端服务（feat/monitor-recorder）

- **类型**：Bug（上游死字段）
- **描述**：`LQRController.__init__` 初始化了 `self.T = 0.0`（line 110）作为「监控」字段，但 `compute()` 从未回写它，导致 `self.T` 恒为 0。本次接入监控时补上了右腿 LQR 轮力矩输出的回写。
- **建议处理方式**：已在本分支修复（compute 内 `self.T = T`）。可顺带核查 PID/MPC 控制器是否也有类似未回写的监控字段。
- **紧急程度**：低

- **类型**：架构洞察
- **描述**：仓库已演进出 `CommandServer` / `StateEstimator` / `BalanceController(lqr/pid/mpc)` 等模块，但 `CLAUDE.md` 的 Architecture 章节仍停留在「VMC + 三路 PID」的旧描述，未提及这些新模块与 UDP 指令服务。
- **建议处理方式**：更新 `CLAUDE.md` 的 Architecture / Control Flow 章节，补充 CommandServer、StateEstimator、各控制器与本次新增的 MonitorServer。
- **紧急程度**：中

- **类型**：重构机会
- **描述**：`StateEstimator.update()` 每个控制周期（~250Hz）无条件 `print()` 大段状态（lines 161-169），刷屏且拖慢控制台。有了网页监控后，这些 print 更显冗余。
- **建议处理方式**：将其改为可开关的 debug 日志，或直接移除，改由 MonitorServer 观测。
- **紧急程度**：低

- **类型**：测试缺口
- **描述**：项目此前无任何测试框架与测试用例（无 pytest、无 tests/ 目录）。本次新增的 `test_monitor_server.py` 是首个测试，使用标准库 unittest 以避免引入新依赖。
- **建议处理方式**：后续若补测控制器/运动学，可统一约定 unittest 或引入 pytest 并在 requirements 中声明。
- **紧急程度**：低
