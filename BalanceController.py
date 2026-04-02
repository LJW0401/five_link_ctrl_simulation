"""
平衡控制器入口 — 支持选择不同控制方案
"""

# 可选控制器类型
CTRL_PID = "pid"
CTRL_LQR = "lqr"


def create_controller(ctrl_type=CTRL_LQR):
    """
    工厂函数：根据类型创建控制器

    参数:
        ctrl_type: "pid" 或 "lqr"
    返回:
        控制器实例（都有 compute(joint_pos, pitch, body_x, body_vx, gyro_y) 接口）
    """
    if ctrl_type == CTRL_PID:
        from PIDController import PIDBalanceController
        return PIDBalanceController()
    elif ctrl_type == CTRL_LQR:
        from LQRController import LQRBalanceController
        return LQRBalanceController()
    else:
        raise ValueError(f"未知控制器类型: {ctrl_type}")
