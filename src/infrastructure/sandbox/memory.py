"""Memory monitoring utilities for sandboxes."""

import threading
import time
from typing import TYPE_CHECKING, Optional

import psutil

if TYPE_CHECKING:
    import subprocess


class MemoryMonitor:
    """
    Monitors a subprocess and kills it if it exceeds the specified memory limit.
    """

    def __init__(self, proc: "subprocess.Popen[str]", max_memory_mb: int):
        self.proc = proc
        self.max_memory_mb = max_memory_mb
        self.exceeded = False
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _monitor(self) -> None:
        try:
            p = psutil.Process(self.proc.pid)
        except psutil.NoSuchProcess:
            return

        while not self.stop_event.is_set() and self.proc.poll() is None:
            try:
                mem_info = p.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                if rss_mb > self.max_memory_mb:
                    self.exceeded = True
                    # Kill the process if it exceeds the memory limit
                    self.proc.kill()
                    break
            except psutil.NoSuchProcess:
                break
            except Exception:
                pass
            time.sleep(0.1)

    def start(self) -> None:
        """Start the memory monitoring thread."""
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the memory monitoring thread."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.2)
