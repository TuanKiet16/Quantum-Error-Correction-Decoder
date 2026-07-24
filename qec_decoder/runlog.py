import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone

from loguru import logger

_CONFIGURED = False


def get_logger():
    global _CONFIGURED
    if not _CONFIGURED:
        os.makedirs("logs", exist_ok=True)
        logger.add("logs/qec_{time}.log", rotation="10 MB", retention=5,
                   enqueue=True, level="INFO")
        _CONFIGURED = True
    return logger


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_run(kind, args, metrics, seed=None, extra=None,
             out_dir="results/runs", start=None, duration_s=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = os.path.join(out_dir, f"{kind}_{stamp}.json")
    rec = {
        "kind": kind,
        "args": args,
        "metrics": metrics,
        "seed": seed,
        "start": start or _now_iso(),
        "end": _now_iso(),
        "duration_s": duration_s if duration_s is not None else 0.0,
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "extra": extra or {},
    }
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    get_logger().info(f"run[{kind}] saved {path} metrics={metrics}")
    return path


class Run:
    def __init__(self, kind, args, seed=None, out_dir="results/runs"):
        self.kind = kind
        self.args = args
        self.seed = seed
        self.out_dir = out_dir
        self.metrics = {}
        self.path = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._start = _now_iso()
        get_logger().info(f"run[{self.kind}] start args={self.args}")
        return self

    def record(self, metrics: dict):
        self.metrics.update(metrics)

    def __exit__(self, exc_type, exc, tb):
        dur = time.perf_counter() - self._t0
        extra = {"error": str(exc)} if exc_type else {}
        self.path = save_run(self.kind, self.args, self.metrics, seed=self.seed,
                             out_dir=self.out_dir, start=self._start,
                             duration_s=dur, extra=extra)
        if exc_type:
            get_logger().error(f"run[{self.kind}] failed: {exc}")
        return False
