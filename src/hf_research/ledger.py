"""File-backed state (atomic JSON) and append-only JSONL ledgers for the paper/live runners."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


class Store:
    def __init__(self, root: Path):
        self.root = root

    def path(self, name: str) -> Path:
        return self.root / name

    def load_state(self) -> dict:
        p = self.path("state.json")
        if not p.exists():
            sys.exit(f"no paper book at {self.root}; run `init` first")
        return json.loads(p.read_text())

    def save_state(self, state: dict) -> None:
        tmp = self.path("state.json.tmp")
        tmp.write_text(json.dumps(state, indent=2, default=float))
        os.replace(tmp, self.path("state.json"))

    def append(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with self.path(name).open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")

    def read(self, name: str) -> pd.DataFrame:
        p = self.path(name)
        if not p.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
