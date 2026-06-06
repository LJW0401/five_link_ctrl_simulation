"""
监控记录后端服务：在 daemon 线程里跑一个 http.server，
接收仿真主循环每个控制周期 push 进来的样本（六变量反馈 + T/Tp 输出），
对外提供网页界面、实时曲线数据、开始/停止记录与 CSV 导出。

职责单一：只负责「收样本 → 缓存 → 经 HTTP 提供给前端」，
不反向读取控制器内部状态——样本由 Simulation 主循环组装后调用 push() 传入，
监控服务与具体控制器解耦。

通道顺序固定（见 CHANNELS）：
    t        仿真时间 (s)
    theta    右腿摆杆夹角 (rad)
    d_theta  右腿摆杆角速度 (rad/s)
    x        机体位移 (m)
    d_x      机体速度 (m/s)
    phi      机体倾角 = -pitch (rad)
    d_phi    机体角速度 (rad/s)
    T        驱动轮力矩输出 (Nm)
    Tp_r     右髋关节力矩输出 (Nm)
    Tp_l     左髋关节力矩输出 (Nm)

依赖：纯标准库（http.server / threading / collections / csv / io / json）。

启动示例（由 Simulation 调用）：
    mon = MonitorServer(host="127.0.0.1", port=8000)
    mon.start()                 # 打印 http://127.0.0.1:8000
    mon.push({...})             # 每个控制周期调用
"""

import csv
import io
import json
import os
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 通道定义：CSV 表头与前端绘图共用同一份顺序，避免错位
CHANNELS = ("t", "theta", "d_theta", "x", "d_x", "phi", "d_phi", "T", "Tp_r", "Tp_l")

# 实时曲线环形缓冲容量（仅供前端显示，与是否记录无关）
LIVE_CAPACITY = 2000

_UI_FILE = os.path.join(os.path.dirname(__file__), "monitor_ui.html")


class MonitorState:
    """线程安全的样本仓库：实时环形缓冲 + 记录缓冲。

    push() 在控制线程（高频）调用，HTTP handler 在服务线程调用，
    所有缓冲访问都用同一把锁保护。
    """

    def __init__(self, capacity=LIVE_CAPACITY):
        self._lock = threading.Lock()
        self._live = deque(maxlen=capacity)  # 实时显示用，自动丢旧
        self._record = []                    # 记录用，start→stop 之间累积
        self._recording = False

    def push(self, sample):
        """接收一条样本（dict，键为 CHANNELS 子集）。
        缺失通道按 0.0 补齐，未知键被忽略，保持仓库纯净。"""
        row = [float(sample.get(k, 0.0)) for k in CHANNELS]
        with self._lock:
            self._live.append(row)
            if self._recording:
                self._record.append(row)

    def start_recording(self):
        """开始一段新记录：清空上次记录缓冲。"""
        with self._lock:
            self._record = []
            self._recording = True

    def stop_recording(self):
        with self._lock:
            self._recording = False

    def status(self):
        with self._lock:
            return {
                "recording": self._recording,
                "recorded": len(self._record),
                "live": len(self._live),
                "channels": list(CHANNELS),
            }

    def live_snapshot(self):
        """返回实时缓冲快照（list[list]），供前端绘图。"""
        with self._lock:
            return {
                "channels": list(CHANNELS),
                "recording": self._recording,
                "samples": list(self._live),
            }

    def export_csv(self):
        """把记录缓冲序列化为 CSV 文本（含表头）。无记录时仅返回表头。"""
        with self._lock:
            rows = list(self._record)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(CHANNELS)
        writer.writerows(rows)
        return buf.getvalue()


def _make_handler(state):
    """闭包工厂：把 MonitorState 注入 handler，避免用类属性传状态。"""

    class _Handler(BaseHTTPRequestHandler):
        # 静默默认的访问日志，避免刷屏仿真控制台
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, content_type="application/json", extra_headers=None):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj, code=200):
            self._send(code, json.dumps(obj), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                try:
                    with open(_UI_FILE, "r", encoding="utf-8") as f:
                        html = f.read()
                except OSError as e:
                    self._send_json({"error": f"无法读取界面文件: {e}"}, 500)
                    return
                self._send(200, html, "text/html; charset=utf-8")
            elif path == "/data":
                self._send_json(state.live_snapshot())
            elif path == "/status":
                self._send_json(state.status())
            elif path == "/export.csv":
                self._send(
                    200,
                    state.export_csv(),
                    "text/csv; charset=utf-8",
                    {"Content-Disposition": 'attachment; filename="monitor_record.csv"'},
                )
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/start":
                state.start_recording()
                self._send_json(state.status())
            elif path == "/stop":
                state.stop_recording()
                self._send_json(state.status())
            else:
                self._send_json({"error": "not found"}, 404)

    return _Handler


class MonitorServer:
    """监控记录服务：持有 MonitorState，并在 daemon 线程跑 HTTP 服务。"""

    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.state = MonitorState()
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(self.state))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        bound_port = self._httpd.server_address[1]  # port=0 时取实际分配端口
        print(f"[MON] 监控界面已启动 → http://{self.host}:{bound_port}  "
              f"（按钮控制开始/停止记录与 CSV 导出）")

    def push(self, sample):
        """供仿真主循环每个控制周期调用，传入一条样本 dict。"""
        self.state.push(sample)

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
