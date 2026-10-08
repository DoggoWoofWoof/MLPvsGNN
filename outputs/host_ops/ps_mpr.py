"""List python processes whose command line names outputs/ (read-only): pid, parent, start time, memory, command."""
import time

import psutil

for p in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time", "memory_info"]):
    try:
        cl = " ".join(p.info["cmdline"] or [])
        if "outputs/" in cl.replace("\\", "/") and "python" in (p.info["name"] or "").lower():
            print(p.info["pid"], p.info["ppid"], time.strftime("%H:%M", time.localtime(p.info["create_time"])),
                  round(p.info["memory_info"].rss / 2 ** 30, 1), cl[-170:])
    except Exception:  # noqa: BLE001
        pass
