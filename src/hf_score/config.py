from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class LabelConfig:
    """How a candidate is turned into a realized outcome.

    Defaults mirror deploy/run_prod_candidate_bch_doge_guarded.sh: exit registry
    artifacts/exit_policy_registry.json, Dynamic Exit V1.2 TP cap of 2.0 ATR on the
    allowlisted asset/sides, shadow exit context defaults (tp/sl mult 1.0, 12-bar time stop).
    """

    exit_registry_path: str = "artifacts/exit_policy_registry.json"
    taker_fee: float = 0.0006
    slippage_bps: float = 3.0
    # Safety cap: a simulated trade still open after this many bars is closed at that bar's close.
    max_horizon_bars: int = 48
    ctx_tp_mult: float = 1.0
    ctx_sl_mult: float = 1.0
    ctx_time_stop_bars: int = 12
    dynamic_exit_tp_cap_atr: float = 2.0
    dynamic_exit_allowlist: tuple[str, ...] = (
        "BTC/USDT:USDT|long",
        "BCH/USDT:USDT|long",
        "XRP/USDT:USDT|long",
        "AVAX/USDT:USDT|short",
        "LINK/USDT:USDT|long",
    )
    # Policy-agnostic forward labels (close-to-close, signed by side, net of round-trip costs).
    forward_horizons: tuple[int, ...] = field(default=(4, 8, 12, 24))
    excursion_horizon_bars: int = 24

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.taker_fee + self.slippage_bps / 10000.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["round_trip_cost"] = self.round_trip_cost
        return d
