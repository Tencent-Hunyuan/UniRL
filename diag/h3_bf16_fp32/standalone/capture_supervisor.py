import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def processes():
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            uid = directory.stat().st_uid
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            env = (directory / "environ").read_bytes().split(b"\0") if uid == os.getuid() else []
            owner = next((v.split(b"=", 1)[1].decode() for v in env if v.startswith(b"CHECKUP_OWNER=")), None)
            result[int(directory.name)] = {"pgid": int(stat[2]), "sid": int(stat[3]),
                                           "start_ticks": int(stat[19]), "owner": owner, "uid": uid}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # 进程可能在读取 /proc 的过程中正常退出。
            continue
    return result


def cleanup(owner):
    signalled = []
    for sig, seconds in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        snapshot = processes()
        groups = {p["pgid"] for p in snapshot.values() if p["owner"] == owner}
        for group in groups:
            # Ray 会另建进程组；只接受组内全部同 UID、同唯一 owner 的组。
            current = processes()
            members = [p for p in current.values() if p["pgid"] == group]
            if not members:
                continue
            if group == os.getpgrp() or any(p["owner"] != owner for p in members):
                raise RuntimeError(f"拒绝停止混合归属进程组 {group}")
            leader_before = snapshot.get(group)
            leader_now = current.get(group)
            if leader_before and leader_now and leader_before["start_ticks"] != leader_now["start_ticks"]:
                raise RuntimeError("PID 已复用，拒绝发送信号")
            try:
                os.killpg(group, sig)
                signalled.append({"pgid": group, "signal": sig.name})
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not any(p["owner"] == owner for p in processes().values()):
                return signalled
            time.sleep(0.5)
    raise RuntimeError("仍有本次 capture 进程；禁止使用全局 ray stop/pkill")


def main():
    parser = argparse.ArgumentParser(description="仅监督本次 replay；原子 capture marker 出现后停止自有进程组")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--groups", type=int, choices=(1, 3), default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/proc/self/environ").exists():
        raise RuntimeError("仅在 Linux 主机运行；本地 CPU 测试不启动 replay")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        raise ValueError("缺少 replay 命令或 timeout 非正")
    args.directory.mkdir(parents=True, exist_ok=False)
    owner = "micro-capture-" + uuid.uuid4().hex
    env = dict(os.environ, CHECKUP_OWNER=owner, H3_DIAG_DUMP_FWD=str(args.directory),
               H3_MICRO_CAPTURE_GROUPS=str(args.groups),
               CHECKUP_SPLITS="full", CHECKUP_OUT_NAME=owner, CHECKUP_STAGE="mcap")  # stage 进 Ray socket 路径，需短
    # 必须在启动前由操作者检查 GPU 进程和授权；本工具不自动挑卡、抢卡。
    child = None
    done = False
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    try:
        with (args.directory / "replay.log").open("x") as stream:
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            (args.directory / "OWNER.json").write_text(json.dumps({"owner": owner, "pid": child.pid,
                 "pgid": child.pid, "supervisor_pid": os.getpid(), "supervisor_pgid": os.getpgrp()}) + "\n")
            while not STOP and time.monotonic() - started < args.timeout:
                marker = args.directory / "CAPTURE_DONE.json"
                if marker.exists():
                    receipt = json.loads(marker.read_text())
                    if (receipt["owner"] != owner or receipt["batch"] != 8
                            or receipt.get("groups", 1) != args.groups):
                        raise RuntimeError("capture marker 归属或 batch 不匹配")
                    if (args.directory / "fwd_rank0.pt").stat().st_size != receipt["bytes"]:
                        raise RuntimeError("capture 大小不同")
                    done = True
                    break
                if child.poll() is not None:
                    raise RuntimeError(f"replay 在 capture 完成前退出，code={child.returncode}")
                time.sleep(1)
    finally:
        groups = cleanup(owner)
        if child is not None:
            child.wait(timeout=10)
        (args.directory / "STOPPED.json").write_text(json.dumps({"capture_complete": done,
            "owned_processes_remaining": 0, "signalled_groups": groups,
            "elapsed_seconds": time.monotonic() - started}) + "\n")
    if not done:
        raise RuntimeError("capture 被中止或超时；保留产物，不自动重试")
    # 完整文件重读验证发生在 GPU 清理后；没有 DONE 就不能把部分文件当作成功。
    import torch
    from standalone_micro import validate_dump
    dump = torch.load(args.directory / "fwd_rank0.pt", weights_only=True, map_location="cpu")
    validate_dump(dump)
    if len(dump.get("groups", [dump])) != args.groups:
        raise ValueError("capture 文件组数与监管器请求不同")
    (args.directory / "VERIFIED").write_text("capture reloaded; batch8; finite outputs/cotangents; own processes stopped\n")
    print("MICRO_CAPTURE_VERIFIED_AND_STOPPED", flush=True)


if __name__ == "__main__":
    main()
