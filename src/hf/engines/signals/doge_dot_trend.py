from __future__ import annotations

import inspect
from dataclasses import replace, is_dataclass
from typing import Any, Iterable, Mapping

from hf.engines.signals.btc_trend_signal import BtcTrendSignalEngine
from hf.engines.signals.dot_trend_signal import DotTrendSignalEngine


def _cfg_params(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg.get("params", {}) or {})
    return dict(getattr(cfg, "params", {}) or {})


def _filter_ctor_kwargs(cls: type, params: dict) -> tuple[dict, dict]:
    params = dict(params or {})

    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return params, {}

    accepted = set()
    has_var_kwargs = False

    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
        elif p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(name)

    if has_var_kwargs:
        return params, {}

    kept = {k: v for k, v in params.items() if k in accepted}
    dropped = {k: v for k, v in params.items() if k not in accepted}
    return kept, dropped


def _allowed_sides(params: dict, default: Iterable[str]) -> set[str]:
    params = dict(params or {})

    use_longs = params.get("use_longs")
    use_shorts = params.get("use_shorts")

    if use_longs is None and use_shorts is None:
        return {str(x).lower() for x in default if str(x).lower() in {"long", "short"}}

    out = set()
    if bool(use_longs):
        out.add("long")
    if bool(use_shorts):
        out.add("short")

    return out or {str(x).lower() for x in default if str(x).lower() in {"long", "short"}}


def _side_of_signal(signal: Any) -> str:
    return str(getattr(signal, "side", "") or "").lower()


def _with_meta(signal: Any, extra_meta: dict) -> Any:
    """
    Attach metadata without depending on mutability.

    Signal is expected to be dataclass-like in this codebase, but this function
    handles mutable objects defensively as well.
    """
    existing = dict(getattr(signal, "meta", {}) or {})
    merged = {**existing, **extra_meta}

    if is_dataclass(signal):
        try:
            return replace(signal, meta=merged)
        except TypeError:
            return signal

    try:
        signal.meta = merged
    except Exception:
        pass

    return signal


class _DedicatedSideFilteredSignalEngine:
    """
    Dedicated asset engine adapter.

    The base signal engines return Dict[str, Signal]. Earlier wrapper versions
    incorrectly treated that output as an iterable of Opportunity objects, which
    dropped all signals. This implementation preserves the real contract:
    generate(...) -> Dict[str, Signal].
    """

    def __init__(
        self,
        *,
        delegate: Any,
        engine_name: str,
        allowed_sides: Iterable[str],
        dropped_params: dict | None = None,
    ) -> None:
        self.delegate = delegate
        self.engine_name = str(engine_name)
        self.allowed_sides = {str(x).lower() for x in allowed_sides if str(x).lower() in {"long", "short"}}
        self.dropped_params = dict(dropped_params or {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def generate(self, *args: Any, **kwargs: Any):
        signals = self.delegate.generate(*args, **kwargs)

        if signals is None:
            return signals

        extra_meta = {
            "dedicated_engine_name": self.engine_name,
            "dedicated_engine_allowed_sides": ",".join(sorted(self.allowed_sides)),
        }
        if self.dropped_params:
            extra_meta["dedicated_engine_dropped_params"] = ",".join(sorted(self.dropped_params.keys()))

        # Normal contract: Dict[str, Signal].
        if isinstance(signals, Mapping):
            out = {}
            for key, signal in signals.items():
                side = _side_of_signal(signal)
                if side not in self.allowed_sides:
                    continue
                out[key] = _with_meta(signal, extra_meta)
            return out

        # Defensive fallback for engines returning list/tuple.
        if isinstance(signals, (list, tuple)):
            out = []
            for signal in signals:
                side = _side_of_signal(signal)
                if side not in self.allowed_sides:
                    continue
                out.append(_with_meta(signal, extra_meta))
            return out

        return signals


class DogeTrendSignalEngine(_DedicatedSideFilteredSignalEngine):
    """
    Dedicated DOGE trend engine.

    It deliberately reuses the generic BTC-style trend logic as the delegate,
    but owns a separate engine name and side filtering seam so DOGE-specific
    behavior can be evolved without polluting BtcTrendSignalEngine.
    """

    def __init__(self, **params: Any) -> None:
        allowed = _allowed_sides(params, default={"long", "short"})

        ctor_params = dict(params)
        ctor_params.pop("use_longs", None)
        ctor_params.pop("use_shorts", None)

        kept, dropped = _filter_ctor_kwargs(BtcTrendSignalEngine, ctor_params)
        delegate = BtcTrendSignalEngine(**kept)

        super().__init__(
            delegate=delegate,
            engine_name="doge_trend_signal",
            allowed_sides=allowed,
            dropped_params=dropped,
        )


class DedicatedDotTrendSignalEngine(_DedicatedSideFilteredSignalEngine):
    """
    Dedicated DOT trend engine.

    It delegates to the existing DotTrendSignalEngine but keeps a separate
    architectural seam and robust side filtering.
    """

    def __init__(self, **params: Any) -> None:
        allowed = _allowed_sides(params, default={"long", "short"})

        # DotTrendSignalEngine natively supports use_longs/use_shorts, but we
        # still filter at the wrapper boundary for consistency and observability.
        kept, dropped = _filter_ctor_kwargs(DotTrendSignalEngine, params)
        delegate = DotTrendSignalEngine(**kept)

        super().__init__(
            delegate=delegate,
            engine_name="dot_trend_signal",
            allowed_sides=allowed,
            dropped_params=dropped,
        )


def make_doge_trend_signal(cfg: Any):
    return DogeTrendSignalEngine(**_cfg_params(cfg))


def make_dot_trend_signal(cfg: Any):
    return DedicatedDotTrendSignalEngine(**_cfg_params(cfg))
