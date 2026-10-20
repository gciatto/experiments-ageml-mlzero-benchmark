"""Run a command and keep resources.json current (stdlib only, runs in the container).

    python supervise.py <resources.json> -- <command...>

Samples the container cgroup (v2) every few seconds so a hard kill still
leaves the last measurement. SIGTERM is forwarded to the child's process group.
"""

import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time

CG = "/sys/fs/cgroup"


def _read(name):
    try:
        with open(os.path.join(CG, name)) as f:
            return f.read()
    except OSError:
        return None


def sample(start):
    cpu = _read("cpu.stat")
    usage = None
    if cpu:
        usage = int(dict(line.split() for line in cpu.splitlines())["usage_usec"]) / 1e6
    peak = _read("memory.peak")
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "elapsed_s": time.time() - start,
        "cgroup_cpu_s": usage,
        "cgroup_peak_bytes": int(peak) if peak else None,
        # rusage covers only reaped children; maxrss is per-process, not summed
        "rusage_children_cpu_s": ru.ru_utime + ru.ru_stime,
        "rusage_children_maxrss_bytes": ru.ru_maxrss * 1024,
        "sampled_at": time.time(),
    }


def write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    out, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1 :]
    start = time.time()
    child = subprocess.Popen(cmd, start_new_session=True)
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(child.pid, signal.SIGTERM))
    done = threading.Event()

    def loop():
        while not done.wait(5):
            write(out, {**sample(start), "final": False})

    threading.Thread(target=loop, daemon=True).start()
    code = child.wait()
    done.set()
    write(out, {**sample(start), "final": True, "exit_code": code})
    sys.exit(code)


if __name__ == "__main__":
    main()
