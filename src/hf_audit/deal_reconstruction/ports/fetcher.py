"""Input port for obtaining raw exchange trade records."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Protocol


RawTradeRecord = Mapping[str, object]


class TradeFetcher(Protocol):
    """Fetch raw trades without interpreting deal semantics."""

    def fetch_trades(
        self,
        *,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> Iterable[RawTradeRecord]:
        """Yield raw exchange trade records in chronological order."""
