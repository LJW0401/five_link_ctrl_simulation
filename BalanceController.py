"""
平衡控制器入口 — 支持选择不同控制方案
"""

# 可选控制器类型
CTRL_PID = "pid"
CTRL_LQR = "lqr"
CTRL_MPC = "mpc"


def create_controller(ctrl_type=CTRL_LQR):
    """
    工厂函数：根据类型创建控制器

    参数:
        ctrl_type: "pid" / "lqr" / "mpc"
    返回:
        控制器实例（都有 compute(imu: IMUData, motors: list[MotorData]) 接口）
    """
    if ctrl_type == CTRL_PID:
        from PIDController import PIDBalanceController
        return PIDBalanceController()
    elif ctrl_type == CTRL_LQR:
        from LQRController import LQRBalanceController
        return LQRBalanceController()
    elif ctrl_type == CTRL_MPC:
        from MPCController import MPCBalanceController
        return MPCBalanceController()
    else:
        raise ValueError(f"未知控制器类型: {ctrl_type}")
