"""
UDP 指令服务器：监听端口接收外部控制参数并缓存最新值。

协议：
  UDP at 127.0.0.1:9000 (可配置)
  payload: JSON，任意字段可省略
  例:
    {"vx": 0.3}              # 目标前进速度 (m/s)
    {"yaw": 0.5}             # 目标 yaw 角 (rad, 绝对)
    {"L0": 0.28}             # 目标腿长 (m)
    {"vx": 0.5, "yaw": 0.3, "L0": 0.25}

发送示例:
    echo '{"vx":0.3,"L0":0.25}' | nc -u -w0 127.0.0.1 9000
    python3 send_command.py --vx 0.3 --L0 0.25
"""

import json
import socket
import threading


SUPPORTED_KEYS = ("vx", "yaw", "L0")


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
        print(f"[CMD] 监听 UDP {self.addr[0]}:{self.addr[1]}，字段: {SUPPORTED_KEYS}")

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
            if not isinstance(msg, dict):
                continue
            filtered = {k: float(v) for k, v in msg.items()
                        if k in SUPPORTED_KEYS and isinstance(v, (int, float))}
            if not filtered:
                continue
            with self._lock:
                self._cmd.update(filtered)
            print(f"[CMD] 收到 {filtered}")

    def snapshot(self):
        """返回当前缓存指令的浅拷贝（线程安全）"""
        with self._lock:
            return dict(self._cmd)


def apply_to_controller(cmd, ctrl):
    """把指令字段映射到控制器目标。控制器若无对应字段则跳过。"""
    if "L0" in cmd and hasattr(ctrl, "L0_target"):
        ctrl.L0_target = cmd["L0"]
    if "yaw" in cmd and hasattr(ctrl, "yaw_target"):
        ctrl.yaw_target = cmd["yaw"]
    if "vx" in cmd and hasattr(ctrl, "v_target"):
        ctrl.v_target = cmd["vx"]
