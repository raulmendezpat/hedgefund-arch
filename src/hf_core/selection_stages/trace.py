from __future__ import annotations

import json
from pathlib import Path


class SelectionTraceWriter:
    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def append(self, rows: list[dict]) -> None:
        if not rows:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
