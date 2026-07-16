"""Enumerations used by the deal-reconstruction domain."""

from __future__ import annotations

from enum import Enum


class PositionSide(str, Enum):
    """Direction of an exchange position."""

    LONG = "long"
    SHORT = "short"


class FillAction(str, Enum):
    """Effect that a fill has on an exchange position."""

    OPEN = "open"
    CLOSE = "close"


class FillOrigin(str, Enum):
    """Known origin of a Bitget fill."""

    API = "api"
    IOS = "ios"
    WEB = "web"
    ANDROID = "android"
    UNKNOWN = "unknown"

    @classmethod
    def from_exchange_value(cls, value: object) -> "FillOrigin":
        """Resolve an exchange-specific source without leaking it elsewhere."""

        normalized = str(value or "").strip().lower()

        aliases = {
            "api": cls.API,
            "ios": cls.IOS,
            "web": cls.WEB,
            "android": cls.ANDROID,
        }
        return aliases.get(normalized, cls.UNKNOWN)


class DealCloseReason(str, Enum):
    """How a reconstructed deal was completed."""

    FULL_CLOSE = "full_close"
    REVERSAL = "reversal"
    END_OF_DATA_OPEN = "end_of_data_open"
