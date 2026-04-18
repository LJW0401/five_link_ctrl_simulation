"""
测试用：向 CommandServer 发送一条 UDP 指令。

用法:
    python3 send_command.py --L0 0.5 --pitch -0.3    # 归一化 ∈ [-1, 1]
    python3 send_command.py --vx 0.3 --yaw 0.5       # 物理量
"""

import argparse
import json
import socket


def _in_range(v):
    if not (-1.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"{v} 超出 [-1, 1]")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    # 所有字段统一归一化 ∈ [-1, 1]
    ap.add_argument("--L0",    type=_in_range, help="腿长 [-1, 1]")
    ap.add_argument("--pitch", type=_in_range, help="机体 pitch [-1, 1]")
    ap.add_argument("--vx",    type=_in_range, help="前进速度 [-1, 1]")
    ap.add_argument("--yaw",   type=_in_range, help="yaw 角 [-1, 1]")
    args = ap.parse_args()

    cmd = {k: v for k, v in (
        ("L0", args.L0), ("pitch", args.pitch),
        ("vx", args.vx), ("yaw", args.yaw),
    ) if v is not None}
    if not cmd:
        print("请至少指定一个字段: --L0 / --pitch / --vx / --yaw")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(cmd).encode("utf-8"), (args.host, args.port))
    print(f"已发送 {cmd} 到 {args.host}:{args.port}")


if __name__ == "__main__":
    main()
