"""
测试用：向 CommandServer 发送一条 UDP 指令。

用法:
    python3 send_command.py --vx 0.3 --L0 0.25
    python3 send_command.py --yaw 0.5
"""

import argparse
import json
import socket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--vx",  type=float)
    ap.add_argument("--yaw", type=float)
    ap.add_argument("--L0",  type=float)
    args = ap.parse_args()

    cmd = {k: v for k, v in (("vx", args.vx), ("yaw", args.yaw), ("L0", args.L0))
           if v is not None}
    if not cmd:
        print("请至少指定一个字段: --vx / --yaw / --L0")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(cmd).encode("utf-8"), (args.host, args.port))
    print(f"已发送 {cmd} 到 {args.host}:{args.port}")


if __name__ == "__main__":
    main()
