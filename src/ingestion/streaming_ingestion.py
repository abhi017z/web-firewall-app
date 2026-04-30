"""
Streaming log ingestion using watchdog file system events.

Provides tail -f like behavior with rotation handling and callback delivery.
"""

from __future__ import annotations

import io
import os
import threading
import time
from typing import Callable, Dict, List, Optional

from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer


class LogStreamHandler(FileSystemEventHandler):
    """Watchdog event handler that tails files and dispatches new lines to a callback."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback = callback
        self._positions: Dict[str, int] = {}
        super().__init__()

    def on_modified(self, event):  # type: ignore[override]
        if isinstance(event, FileModifiedEvent) and not event.is_directory:
            self._read_new_lines(event.src_path)

    def on_created(self, event):  # type: ignore[override]
        if isinstance(event, FileCreatedEvent) and not event.is_directory:
            # New file: initialize position
            self._positions[event.src_path] = 0
            self._read_new_lines(event.src_path)

    def on_moved(self, event):  # type: ignore[override]
        if isinstance(event, FileMovedEvent) and not event.is_directory:
            # Handle rotation: old file moved, start tailing new path if exists
            old = event.src_path
            new = event.dest_path
            self._positions.pop(old, None)
            self._positions[new] = 0
            if os.path.exists(new):
                self._read_new_lines(new)

    def _read_new_lines(self, path: str) -> None:
        try:
            last_pos = self._positions.get(path, 0)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(last_pos)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        self.callback(line)
                self._positions[path] = f.tell()
        except FileNotFoundError:
            # File rotated away
            self._positions.pop(path, None)


_observer: Optional[Observer] = None


def start_streaming(log_paths: List[str], callback: Callable[[str], None]) -> None:
    """Start streaming new log lines from multiple files/directories.

    Args:
        log_paths: List of file or directory paths to watch
        callback: Function receiving each new log line as a string
    """
    global _observer
    if _observer is not None:
        return
    handler = LogStreamHandler(callback)
    observer = Observer()
    for p in log_paths:
        watch_path = p if os.path.isdir(p) else os.path.dirname(p) or "."
        observer.schedule(handler, watch_path, recursive=False)
        # Initialize position for existing file
        if os.path.isfile(p):
            handler._positions[p] = os.path.getsize(p)
    observer.daemon = True
    observer.start()
    _observer = observer


def stop_streaming() -> None:
    """Stop streaming and shut down the observer."""
    global _observer
    if _observer is None:
        return
    _observer.stop()
    _observer.join(timeout=5)
    _observer = None


__all__ = [
    "LogStreamHandler",
    "start_streaming",
    "stop_streaming",
]


