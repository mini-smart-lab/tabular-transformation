from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any


class EpisodeLogger:
    def __init__(self, path: Path | None = None, logger: logging.Logger | None = None) -> None:
        self.path = path
        self.logger = logger or logging.getLogger("tabular_benchmark.episode")
        self._lock = Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **payload: Any) -> None:
        entry = {"event": event, **payload}
        self.logger.info("benchmark event", extra={"benchmark_event": entry})
        if self.path is not None:
            line = json.dumps(entry, ensure_ascii=False, default=str)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
