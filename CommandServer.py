"""
UDP 指令服务器：监听端口接收外部控制参数并缓存最新值。

协议：
  UDP at 127.0.0.1:9000 (可配置)
  payload: JSON，任意字段可省略
  **所有字段输入范围统一为 [-1, 1]**，内部解归一化为物理量：

    L0     → [L0_MIN, L0_MAX]      腿长 (m)
    pitch  → [-PITCH_MAX, +PITCH_MAX]  机体 pitch 目标 (rad)
    vx     → [-VX_MAX, +VX_MAX]    前进速度 (m/s)
    yaw    → [-YAW_MAX, +YAW_MAX]  yaw 角 (rad, 绝对)

  越界输入将被拒绝并打印警告。

  例:
    {"L0": 0.0}                   # → 0.25 m（中点）
    {"L0": 1.0, "pitch": -0.5}    # → 0.40 m, -0.1 rad
    {"vx": 0.3, "yaw": -0.5}      # → 0.3 m/s, -π/2 rad

发送示例:
    echo '{"L0":0.5,"pitch":-0.3}' | nc -u -w0 127.0.0.1 9000
    python3 send_command.py --L0 0.5 --pitch -0.3
"""

import json
import math
import socket
import threading


# ========== 各字段物理范围 ==========
L0_MIN = 0.10          # 腿长下限 (m)
L0_MAX = 0.40          # 腿长上限 (m)
PITCH_MAX = 0.20       # pitch 目标最大幅度 (rad)
VX_MAX = 3.0           # 前进速度最大幅度 (m/s)
YAW_MAX = math.pi      # yaw 目标最大幅度 (rad)

SUPPORTED_KEYS = ("L0", "pitch", "vx", "yaw")


def _validate(msg):
    """解析并校验 UDP 消息。所有字段必须 ∈ [-1, 1]。
    返回 dict（已过滤非法字段），非 dict/无合法字段返回 None。"""
    if not isinstance(msg, dict):
        return None
    out = {}
    for k, v in msg.items():
        if k not in SUPPORTED_KEYS:
            print(f"[CMD] 忽略未知字段 {k!r}")
            continue
        if not isinstance(v, (int, float)):
            print(f"[CMD] 字段 {k!r} 类型错误（需 number）: {v!r}")
            continue
        fv = float(v)
        if not (-1.0 <= fv <= 1.0):
            print(f"[CMD] 字段 {k!r}={fv:+.3f} 越界（需 ∈ [-1, 1]），已拒绝")
            continue
        out[k] = fv
    return out or None


class CommandServer:
    def __init__(self, host="127.0.0.1", port=9000):
        self.addr = (host, port)
        self._cmd = {}
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(self.addr)
        self._sock.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        print(f"[CMD] 监听 UDP {self.addr[0]}:{self.addr[1]}，"
              f"字段 {SUPPORTED_KEYS} ∈ [-1, 1]")

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"[CMD] 解析失败: {e}")
                continue

            filtered = _validate(msg)
            if filtered is None:
                continue
            with self._lock:
                self._cmd.update(filtered)
            print(f"[CMD] 收到 {filtered}")

    def snapshot(self):
        """返回当前缓存指令的浅拷贝（线程安全）"""
        with self._lock:
            return dict(self._cmd)


# ========== 解归一化：-1~1 → 物理量 ==========

def _denorm_L0(n):
    mid = 0.5 * (L0_MIN + L0_MAX)
    half = 0.5 * (L0_MAX - L0_MIN)
    return mid + half * n

def _denorm_pitch(n):
    return PITCH_MAX * n

def _denorm_vx(n):
    return VX_MAX * n

def _denorm_yaw(n):
    return YAW_MAX * n


def apply_to_controller(cmd, ctrl):
    """
    把指令字段映射到控制器目标。控制器若无对应字段则跳过。
    所有字段在此处解归一化为物理值。
    """
    if "L0" in cmd and hasattr(ctrl, "L0_target"):
        ctrl.L0_target = _denorm_L0(cmd["L0"])
    if "pitch" in cmd and hasattr(ctrl, "pitch_target"):
        ctrl.pitch_target = _denorm_pitch(cmd["pitch"])
    if "yaw" in cmd and hasattr(ctrl, "yaw_target"):
        ctrl.yaw_target = _denorm_yaw(cmd["yaw"])
    if "vx" in cmd and hasattr(ctrl, "v_target"):
        ctrl.v_target = _denorm_vx(cmd["vx"])
