import math
from math import sin,cos

class leg_VMC:
    """
    虚拟机械腿参数类
    """
    def __init__(self, leg_params=None):
        # 长度参数，单位为m
        if leg_params is not None:
            self.l1 = leg_params["l1"]
            self.l2 = leg_params["l2"]
            self.l3 = leg_params["l3"]
            self.l4 = leg_params["l4"]
            self.l5 = leg_params["l5"]
        else:
            self.l5 = 0.0  # AE距离
            self.l1 = 0.215
            self.l4 = 0.215
            self.l2 = 0.258
            self.l3 = 0.258
        
        self.phi2 = 0.0
        self.phi3 = 0.0
        self.phi1 = 0.0
        self.phi4 = 0.0
        
        # 雅可比矩阵系数
        self.j11 = 0.0
        self.j12 = 0.0
        self.j21 = 0.0
        self.j22 = 0.0
        
        # 扭矩设置值（2个元素的列表）
        self.torque_set = [0.0, 0.0]
        
        # 力和扭矩参数
        self.F0 = 0.0
        self.Tp = 0.0
        self.F02 = 0.0
        
        # L0相关参数
        self.L0 = 0.0
        self.Phi0 = 0.0

        # 标志位
        self.first_flag = 0

    def calc_forward_kinematics(self, phi1, phi4):
        """
        给定关节角度 phi1 和 phi4，计算虚拟腿参数 L0 和 phi0
        """
        self.phi1 = phi1
        self.phi4 = phi4

        # 计算B点和D点的坐标
        YD = self.l4 * math.sin(phi4)  # D点y坐标
        YB = self.l1 * math.sin(phi1)  # B点y坐标
        XD = self.l5 + self.l4 * math.cos(phi4)  # D点x坐标
        XB = self.l1 * math.cos(phi1)  # B点x坐标
        
        # 计算BD距离
        lBD = math.sqrt((XD - XB)**2 + (YD - YB)**2)
        
        # 计算phi2和phi3（基于几何约束）
        A0 = 2 * self.l2 * (XD - XB)
        B0 = 2 * self.l2 * (YD - YB)
        C0 = self.l2**2 + lBD**2 - self.l3**2
        
        discriminant = A0**2 + B0**2 - C0**2
        if discriminant < 0:
            raise ValueError("无解：给定的关节角度超出工作空间")
        
        self.phi2 = 2 * math.atan2(B0 + math.sqrt(discriminant), A0 + C0)
        self.phi3 = math.atan2(YB - YD + self.l2 * math.sin(self.phi2), XB - XD + self.l2 * math.cos(self.phi2))

        XC = XB + self.l2 * math.cos(self.phi2)
        YC = YB + self.l2 * math.sin(self.phi2)

        self.L0 = math.sqrt((XC - self.l5/2.0)**2 + YC**2)
        self.Phi0 = math.atan2(YC, (XC - self.l5/2.0))

        return self.L0, self.Phi0

    def calc_inverse_kinematics(self, L0, phi0):
        """
        逆运动学：给定虚拟腿参数 (L0, phi0)，求解关节角度 (phi1, phi4)

        参数:
            L0:   虚拟腿长度 (m)
            phi0: 虚拟腿极角 (rad)，以 (l5/2, 0) 为极点

        返回:
            (phi1, phi4): 两个驱动关节角度 (rad)
            如果无解返回 None
        """
        # 末端 C 坐标（以 A 为原点，M=(l5/2,0) 为虚拟腿极点）
        Cx = self.l5 / 2.0 + L0 * math.cos(phi0)
        Cy = L0 * math.sin(phi0)

        # 左三角形 A(0,0)-B-C：AB=l1, BC=l2
        Lca2 = Cx * Cx + Cy * Cy
        Lca = math.sqrt(Lca2)
        cos_b1 = (self.l1 * self.l1 + Lca2 - self.l2 * self.l2) / (2.0 * self.l1 * Lca)

        # 右三角形 E(l5,0)-D-C：ED=l4, DC=l3
        ECx = Cx - self.l5
        Lce2 = ECx * ECx + Cy * Cy
        Lce = math.sqrt(Lce2)
        cos_b4 = (self.l4 * self.l4 + Lce2 - self.l3 * self.l3) / (2.0 * self.l4 * Lce)

        if abs(cos_b1) > 1.0 or abs(cos_b4) > 1.0:
            return None

        # B 取 AC 上侧、D 取 EC 下侧（机构装配模式）
        phi1 = math.atan2(Cy, Cx)  + math.acos(cos_b1)
        phi4 = math.atan2(Cy, ECx) - math.acos(cos_b4)

        return (phi1, phi4)

    def calc_jacobian(self):
        """
        计算雅可比矩阵系数 j11, j12, j21, j22
        """
        sin_phi3_phi2 = math.sin(self.phi3 - self.phi2)
        if abs(sin_phi3_phi2) < 1e-6:
            raise ValueError("sin(phi3 - phi2) 太小，可能导致数值不稳定")

        self.j11 = (self.l1 * math.sin(self.Phi0 - self.phi3) * math.sin(self.phi1 - self.phi2)) / sin_phi3_phi2
        self.j12 = (self.l1 * math.cos(self.Phi0 - self.phi3) * math.sin(self.phi1 - self.phi2)) / (self.L0 * sin_phi3_phi2)
        self.j21 = (self.l4 * math.sin(self.Phi0 - self.phi2) * math.sin(self.phi3 - self.phi4)) / sin_phi3_phi2
        self.j22 = (self.l4 * math.cos(self.Phi0 - self.phi2) * math.sin(self.phi3 - self.phi4)) / (self.L0 * sin_phi3_phi2)

    def calc_torque(self, F0, Tp):
        """
        计算关节力矩

        参数:
            F0: 支持力 (N)
            Tp: 扭转力 (Nm)

        返回:
            (torque1, torque4): 两个驱动关节的力矩 (Nm)
        """
        self.calc_jacobian()
        torque1 = self.j11 * F0 + self.j12 * Tp
        torque4 = self.j21 * F0 + self.j22 * Tp
        
        self.torque_set[0] = torque1
        self.torque_set[1] = torque4
        
        return torque1, torque4

    def calc(self, phi1, phi4, F0, Tp):
        """
        计算关节力矩的主函数，包含正运动学和雅可比计算

        参数:
            phi1: 后关节角度 (rad)
            phi4: 前关节角度 (rad)
            F0: 支持力 (N)
            Tp: 扭转力 (Nm)

        返回:
            (torque1, torque4): 两个驱动关节的力矩 (Nm)
        """
        self.calc_forward_kinematics(phi1, phi4)
        
        return self.calc_torque(F0, Tp)
