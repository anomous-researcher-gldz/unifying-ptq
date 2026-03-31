# persistent_avg.py
import json, os, tempfile, fcntl
from pathlib import Path

def _load_state(p: str):
    try:
        with open(p, "r") as f:
            d = json.load(f)
        return int(d.get("n", 0)), float(d.get("sum", 0.0))
    except FileNotFoundError:
        return 0, 0.0

def _atomic_write_json(p: str, obj: dict):
    d = os.path.dirname(p) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".avg.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)  # atomic on POSIX
    finally:
        try: os.remove(tmp)
        except FileNotFoundError: pass

def _lock(path: str):
    lock_path = f"{path}.lock"
    f = open(lock_path, "a+")
    fcntl.flock(f, fcntl.LOCK_EX)   # blocks until lock acquired
    return f                         # caller must close() to release

def get_avg(state_path="global_avg.json"):
    n, s = _load_state(state_path)
    mu = (s / n) if n else 0.0
    return n, mu

def peek_after_add(x: float, state_path="global_avg.json"):
    n, s = _load_state(state_path)
    n2, s2 = n + 1, s + float(x)
    return n2, (s2 / n2)

def update_avg(x: float, state_path="global_avg.json"):
    """Add value x, persist (n,sum), return (n_new, mean_new)."""
    lockf = _lock(state_path)
    try:
        n, s = _load_state(state_path)
        n2, s2 = n + 1, s + float(x)
        _atomic_write_json(state_path, {"n": n2, "sum": s2, "mean": s2 / n2})
        return n2, (s2 / n2)
    finally:
        lockf.close()

def reset_avg(state_path="global_avg.json"):
    lockf = _lock(state_path)
    try:
        _atomic_write_json(state_path, {"n": 0, "sum": 0.0, "mean": 0.0})
    finally:
        lockf.close()
