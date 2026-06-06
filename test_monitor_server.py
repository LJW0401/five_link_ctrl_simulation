"""
MonitorServer 测试：MonitorState 数据逻辑 + HTTP 端点。

Smoke：push→实时快照、记录→导出 CSV、HTTP 各端点连通。
异常/边界：缺失通道、未知键、未记录时不入库、环形缓冲溢出、并发 push、空记录导出。

纯标准库（unittest / urllib / threading），无需 pytest。
运行：python3 -m unittest test_monitor_server -v
"""

import csv
import io
import threading
import unittest
import urllib.request
import urllib.error

from MonitorServer import MonitorServer, MonitorState, CHANNELS, LIVE_CAPACITY


def _full_sample(t=0.0, v=1.0):
    s = {k: v for k in CHANNELS}
    s["t"] = t
    return s


class TestMonitorState(unittest.TestCase):
    # ---------- Smoke ----------
    def test_push_appears_in_live(self):
        st = MonitorState()
        st.push(_full_sample(t=0.5))
        snap = st.live_snapshot()
        self.assertEqual(len(snap["samples"]), 1)
        self.assertEqual(snap["channels"], list(CHANNELS))
        self.assertEqual(snap["samples"][0][0], 0.5)  # t 在第 0 列

    def test_record_then_export(self):
        st = MonitorState()
        st.start_recording()
        st.push(_full_sample(t=0.0, v=2.0))
        st.push(_full_sample(t=0.004, v=3.0))
        st.stop_recording()
        rows = list(csv.reader(io.StringIO(st.export_csv())))
        self.assertEqual(rows[0], list(CHANNELS))      # 表头
        self.assertEqual(len(rows), 3)                  # 表头 + 2 行
        self.assertEqual(rows[1][1], "2.0")             # theta 列

    # ---------- 异常 / 边界 ----------
    def test_missing_channels_filled_zero(self):
        # 非法输入：只给部分通道，其余补 0.0
        st = MonitorState()
        st.push({"t": 1.0, "theta": 5.0})
        row = st.live_snapshot()["samples"][0]
        self.assertEqual(row[CHANNELS.index("theta")], 5.0)
        self.assertEqual(row[CHANNELS.index("Tp_l")], 0.0)

    def test_unknown_key_ignored(self):
        # 非法输入：未知键不应进入仓库（行长度恒等于通道数）
        st = MonitorState()
        st.push({"t": 1.0, "garbage": 99.0})
        row = st.live_snapshot()["samples"][0]
        self.assertEqual(len(row), len(CHANNELS))

    def test_not_recording_keeps_record_empty(self):
        # 状态边界：未 start 时 push 只进实时缓冲，不进记录
        st = MonitorState()
        st.push(_full_sample())
        self.assertEqual(st.status()["recorded"], 0)
        self.assertEqual(st.status()["live"], 1)

    def test_export_empty_record_header_only(self):
        # 边界值：无记录导出，仅表头
        st = MonitorState()
        rows = list(csv.reader(io.StringIO(st.export_csv())))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], list(CHANNELS))

    def test_live_ring_buffer_overflow(self):
        # 边界值：超过容量后丢弃最旧样本
        st = MonitorState()
        for i in range(LIVE_CAPACITY + 50):
            st.push(_full_sample(t=float(i)))
        snap = st.live_snapshot()
        self.assertEqual(len(snap["samples"]), LIVE_CAPACITY)
        self.assertEqual(snap["samples"][0][0], 50.0)   # 最旧的 0..49 已被丢弃

    def test_concurrent_push(self):
        # 并发竞态：多线程同时 push，记录计数应精确无丢失/无崩溃
        st = MonitorState()
        st.start_recording()
        n_threads, per = 8, 500

        def worker():
            for _ in range(per):
                st.push(_full_sample())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        st.stop_recording()
        self.assertEqual(st.status()["recorded"], n_threads * per)

    def test_start_clears_previous_record(self):
        # 状态边界：重新 start 应清空上一段记录
        st = MonitorState()
        st.start_recording()
        st.push(_full_sample())
        st.stop_recording()
        st.start_recording()
        self.assertEqual(st.status()["recorded"], 0)


class TestMonitorHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = MonitorServer(host="127.0.0.1", port=0)  # 端口 0 → 自动分配空闲端口
        cls.server.start()
        cls.port = cls.server._httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=3) as r:
            return r.status, r.headers, r.read()

    def _post(self, path):
        req = urllib.request.Request(self.base + path, method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, r.read()

    # ---------- Smoke ----------
    def test_index_served(self):
        status, _, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<canvas", body)

    def test_record_cycle_over_http(self):
        self._post("/start")
        self.server.push(_full_sample(v=7.0))
        self._post("/stop")
        status, headers, body = self._get("/export.csv")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers.get("Content-Disposition", ""))
        rows = list(csv.reader(io.StringIO(body.decode("utf-8"))))
        self.assertGreaterEqual(len(rows), 2)  # 表头 + 至少 1 行

    def test_data_endpoint(self):
        status, _, body = self._get("/data")
        self.assertEqual(status, 200)
        import json
        obj = json.loads(body)
        self.assertIn("samples", obj)
        self.assertEqual(obj["channels"], list(CHANNELS))

    # ---------- 异常 ----------
    def test_unknown_path_404(self):
        # 异常恢复：未知路径返回 404 而非崩溃
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/nope")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
