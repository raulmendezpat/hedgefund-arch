from __future__ import annotations

import pandas as pd

from .contracts import SelectionContext


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


class BestPerSymbolStage:
    def __init__(self, score_field: str = "policy_score"):
        self.score_field = str(score_field or "policy_score")

    def apply(self, ctx: SelectionContext) -> SelectionContext:
        if not ctx.rows:
            ctx.selected_idx = []
            return ctx

        keep_set = set(int(x) for x in list(ctx.selected_idx or []))

        rows = []
        for r in ctx.rows:
            if int(r.idx) not in keep_set:
                continue

            selection_meta = getattr(r, "selection_meta", None)
            if not isinstance(selection_meta, dict):
                selection_meta = {}

            alpha_meta = dict(selection_meta.get("alpha_selection", {}) or {})
            alpha_score = _safe_float(alpha_meta.get("score", 0.0), 0.0)

            rows.append(
                {
                    "idx": int(r.idx),
                    "ts": int(r.ts),
                    "symbol": str(r.symbol),
                    "strategy_id": str(r.strategy_id),
                    "side": str(r.side),
                    "policy_score": float(r.policy_score),
                    "alpha_score": float(alpha_score),
                    "p_win": float(r.p_win),
                    "post_ml_score": float(r.post_ml_score),
                    "competitive_score": float(r.competitive_score),
                }
            )

        if not rows:
            ctx.selected_idx = []
            return ctx

        df = pd.DataFrame(rows)

        score_col = self.score_field
        if score_col not in df.columns:
            score_col = "policy_score"

        df["_sort_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)

        kept = []
        trace_rows = []

        for symbol, g in df.groupby("symbol", sort=False):
            g = g.sort_values(["_sort_score"], ascending=[False]).copy()
            winner = g.iloc[0]
            kept.append(int(winner["idx"]))

            for _, row in g.iterrows():
                trace_rows.append(
                    {
                        "stage": "best_per_symbol",
                        "ts": int(row["ts"]),
                        "symbol": str(row["symbol"]),
                        "strategy_id": str(row["strategy_id"]),
                        "side": str(row["side"]),
                        "idx": int(row["idx"]),
                        "score_field": str(score_col),
                        "score_value": float(row["_sort_score"]),
                        "winner": bool(int(row["idx"]) == int(winner["idx"])),
                    }
                )

        ctx.selected_idx = kept
        ctx.trace_rows.extend(trace_rows)
        ctx.meta["best_per_symbol_kept"] = int(len(kept))
        return ctx
