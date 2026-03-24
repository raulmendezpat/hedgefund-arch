from __future__ import annotations

import pandas as pd

from .contracts import SelectionContext, SelectionRow
from .config import resolve_profile_config


def _pct_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if len(s) <= 1:
        return pd.Series([1.0] * len(s), index=s.index, dtype=float)
    return s.rank(method="average", pct=True).astype(float)


class AssetGateStage:
    def __init__(self, cfg: dict, profile: str = "research"):
        self.cfg = cfg
        self.profile = str(profile)

    def apply(self, ctx: SelectionContext) -> SelectionContext:
        if not ctx.rows:
            ctx.selected_idx = []
            return ctx

        df = pd.DataFrame([
            {
                "idx": r.idx,
                "ts": r.ts,
                "symbol": r.symbol,
                "strategy_id": r.strategy_id,
                "side": r.side,
                "p_win": float(r.p_win),
                "post_ml_score": float(r.post_ml_score),
                "competitive_score": float(r.competitive_score),
                "policy_score": float(r.policy_score),
                "accept_in": bool(r.accept_in),
            }
            for r in ctx.rows
        ])

        df = df[df["accept_in"].eq(True) & df["side"].isin(["long", "short"])].copy()
        if df.empty:
            ctx.selected_idx = []
            return ctx

        keep_parts = []
        trace_rows = []

        for (symbol, side), g in df.groupby(["symbol", "side"], sort=False):
            g = g.copy()
            rcfg = resolve_profile_config(self.cfg, symbol=symbol, side=side, profile=self.profile)
            ag = dict(rcfg.get("asset_gate", {}) or {})

            gate_mode = str(ag.get("mode", "strict") or "strict").lower()

            g["pwin_rank"] = _pct_rank(g["p_win"])
            g["postml_rank"] = _pct_rank(g["post_ml_score"])
            g["competitive_rank"] = _pct_rank(g["competitive_score"])

            min_pwin_strong = float(ag.get("min_pwin_strong", 0.80))
            min_pwin_contextual = float(ag.get("min_pwin_contextual", 0.70))
            min_policy_score = float(ag.get("min_policy_score", 0.0))
            min_pwin_rank = float(ag.get("min_pwin_rank", 0.70))
            min_postml_rank = float(ag.get("min_postml_rank", 0.60))
            min_competitive_rank = float(ag.get("min_competitive_rank", 0.55))

            strong_abs = g["p_win"] >= min_pwin_strong
            contextual = (
                (g["p_win"] >= min_pwin_contextual) &
                (g["policy_score"] >= min_policy_score) &
                (g["pwin_rank"] >= min_pwin_rank) &
                (g["postml_rank"] >= min_postml_rank) &
                (g["competitive_rank"] >= min_competitive_rank)
            )

            g["stage_asset_gate_pass"] = strong_abs | contextual
            keep_parts.append(g)

            for _, row in g.iterrows():
                trace_rows.append({
                    "stage": "asset_gate",
                    "ts": int(row["ts"]),
                    "symbol": str(symbol),
                    "side": str(side),
                    "strategy_id": str(row["strategy_id"]),
                    "idx": int(row["idx"]),
                    "p_win": float(row["p_win"]),
                    "policy_score": float(row["policy_score"]),
                    "pwin_rank": float(row["pwin_rank"]),
                    "postml_rank": float(row["postml_rank"]),
                    "competitive_rank": float(row["competitive_rank"]),
                    "pass": bool(row["stage_asset_gate_pass"]),
                    "min_pwin_strong": min_pwin_strong,
                    "min_pwin_contextual": min_pwin_contextual,
                    "min_pwin_rank": min_pwin_rank,
                    "min_postml_rank": min_postml_rank,
                    "min_competitive_rank": min_competitive_rank,
                })

        out = pd.concat(keep_parts, ignore_index=False)

        # modo observe_only: no filtrar, solo observar
        if gate_mode in {"observe_only", "bypass", "off"}:
            ctx.selected_idx = out["idx"].astype(int).tolist()
        else:
            ctx.selected_idx = out.loc[out["stage_asset_gate_pass"].eq(True), "idx"].astype(int).tolist()

        ctx.trace_rows.extend(trace_rows)
        ctx.meta["asset_gate_kept"] = int(len(ctx.selected_idx))
        ctx.meta["asset_gate_mode"] = str(gate_mode)
        return ctx
