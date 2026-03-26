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

        row_by_idx = {int(r.idx): r for r in ctx.rows}

        for symbol, g in df.groupby("symbol", sort=False):
            g = g.sort_values(["_sort_score", "alpha_score", "p_win", "policy_score"], ascending=[False, False, False, False]).copy()
            g["_rank"] = range(1, len(g) + 1)
            winner = g.iloc[0]
            runner_up = g.iloc[1] if len(g) > 1 else None
            winner_score = float(winner["_sort_score"])
            runner_up_score = float(runner_up["_sort_score"]) if runner_up is not None else 0.0
            win_margin = float(winner_score - runner_up_score)
            competitor_count = int(len(g))

            kept.append(int(winner["idx"]))

            for _, row in g.iterrows():
                row_idx = int(row["idx"])
                is_winner = bool(row_idx == int(winner["idx"]))
                rank = int(row["_rank"])
                row_obj = row_by_idx.get(row_idx)
                if row_obj is not None:
                    selection_meta = getattr(row_obj, "selection_meta", None)
                    if not isinstance(selection_meta, dict):
                        selection_meta = {}
                        row_obj.selection_meta = selection_meta
                    stage_bucket = dict(selection_meta.get("best_per_symbol", {}) or {})
                    stage_bucket.update(
                        {
                            "winner": bool(is_winner),
                            "kept": bool(is_winner),
                            "rank": int(rank),
                            "competitor_count": int(competitor_count),
                            "score_field": str(score_col),
                            "score_value": float(row["_sort_score"]),
                            "winner_idx": int(winner["idx"]),
                            "winner_side": str(winner["side"]),
                            "winner_strategy_id": str(winner["strategy_id"]),
                            "winner_score": float(winner_score),
                            "runner_up_idx": int(runner_up["idx"]) if runner_up is not None else None,
                            "runner_up_score": float(runner_up_score),
                            "win_margin": float(win_margin),
                            "reason": "top_score_for_symbol",
                        }
                    )
                    selection_meta["best_per_symbol"] = stage_bucket

                trace_rows.append(
                    {
                        "stage": "best_per_symbol",
                        "ts": int(row["ts"]),
                        "symbol": str(row["symbol"]),
                        "strategy_id": str(row["strategy_id"]),
                        "side": str(row["side"]),
                        "idx": int(row_idx),
                        "score_field": str(score_col),
                        "score_value": float(row["_sort_score"]),
                        "winner": bool(is_winner),
                        "kept": bool(is_winner),
                        "rank": int(rank),
                        "competitor_count": int(competitor_count),
                        "winner_idx": int(winner["idx"]),
                        "winner_side": str(winner["side"]),
                        "winner_strategy_id": str(winner["strategy_id"]),
                        "winner_score": float(winner_score),
                        "runner_up_idx": int(runner_up["idx"]) if runner_up is not None else None,
                        "runner_up_score": float(runner_up_score),
                        "win_margin": float(win_margin),
                        "reason": "top_score_for_symbol",
                    }
                )

        ctx.selected_idx = kept
        ctx.trace_rows.extend(trace_rows)
        ctx.meta["best_per_symbol_kept"] = int(len(kept))
        return ctx
