"""Trend following and stock momentum on Bitget's traditional-asset perpetuals
(pre-registered in docs/plans/tradfi_trend_hypothesis.md).

  PYTHONPATH=src python scripts/run_tradfi_study.py            # downloads proxies into the cache
  PYTHONPATH=src python scripts/run_tradfi_study.py --no-fetch
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.engine import backtest_weights, summarize
from hf_research.tradfi import bitget_rate_drag, load_closes, update_yahoo
from hf_research.tsmom import TsmomConfig, tsmom_targets
from hf_research.xs_momentum import MomentumConfig, target_weights

# Frozen inputs measured on 2026-10-31 (artifacts/research/tradfi_v1/measured_20261031/).
# perpetual: (proxy, asset class, 10k impact bp, funding %/yr; longs pay when positive)
MACRO = {
    "SPY": ("SPY", "equities", 1.3, 3.4),
    "QQQ": ("QQQ", "equities", 1.4, 6.3),
    "IWM": ("IWM", "equities", 11.4, 0.1),
    "EWJ": ("EWJ", "equities", 13.5, 0.0),
    "EWY": ("EWY", "equities", 3.6, 3.6),
    "EWT": ("EWT", "equities", 18.4, -0.7),
    "HSI": ("^HSI", "equities", 2.3, 4.3),
    "TLT": ("TLT", "bonds", 6.1, 2.6),
    "XAU": ("GLD", "metals", 0.0, 12.0),
    "XAG": ("SLV", "metals", 1.6, 12.4),
    "XPT": ("PPLT", "metals", 4.2, 19.3),
    "XPD": ("PALL", "metals", 18.0, 24.8),
    "CL": ("USO", "energy", 1.0, 3.7),
    "BZ": ("BNO", "energy", 1.1, 9.7),
    "NATGAS": ("UNG", "energy", 5.5, 32.2),
    "COPPER": ("CPER", "energy", 8.8, 5.1),
    "EURUSD": ("EURUSD=X", "fx", 3.2, -0.7),
    "GBPUSD": ("GBPUSD=X", "fx", 4.3, -6.8),
    "USDBRL": ("BRL=X", "fx", 7.0, 2.6),
}
STOCKS = ("INTC MU MSTR GOOGL NVDA AMZN TSLA AMD MRVL AAPL MSFT AVGO WDC COHR ORCL GLW TSM QCOM ASML NKE "
          "FTNT BP SBUX AKAM PBR MDT DHR EBAY HD").split()
STOCK_COSTS = "artifacts/research/tradfi_v1/measured_20261031/stocks_liquidity.csv"

FEE = 0.0006
MIN_SLIP_BPS = 2.0
START, SPLIT, END = "2008-01-01", "2018-01-01", "2026-10-31"
DECISION_WEEKDAY = 4  # Friday close; filled at the next trading day's close
H1 = TsmomConfig(lookbacks=(30, 90, 365), vol_window=60, asset_vol_target=0.03, max_weight=0.5, band=0.01)
H1_MAX_GROSS = 3.0
H1_CRITERIA = {"oos_sharpe_min": 0.5, "is_sharpe_min": 0.3, "oos_max_dd_min": -0.20,
               "oos_sharpe_ex_best_class_min": 0.25, "bootstrap_p_max": 0.10}
H2 = MomentumConfig(lookback_days=365, skip_days=30, select_frac=0.2, min_assets=8, band=0.02)
H2_EXPLORATORY = replace(H2, lookback_days=30, skip_days=1)
H2_CRITERIA = {"oos_sharpe_min": 0.5, "is_sharpe_min": 0.3, "oos_max_dd_min": -0.30, "oos_sharpe_ex_top3_min": 0.25}
COMBO = {"scale_start": "2021-03-01", "scale_end": "2024-01-01", "eval_start": "2024-01-01", "sharpe_gain_min": 0.15}
CRYPTO_DAILY = "artifacts/research/tsmom_btc_eth_v1/daily_net.csv"
GRID = {"lookbacks": [(30,), (90,), (365,), (30, 90, 365), (20, 60, 120)], "weekly": [True, False], "cost_mult": [1.0, 2.0]}


def build_panels(closes: pd.DataFrame, slippage_bps: pd.Series, funding_pct: pd.Series) -> dict[str, pd.DataFrame]:
    """Calendar-day UTC panels for the engine. Row d: signal = last close <= d; fill = close of the
    first trading day after d (one-day lag); funding = constant annual rate / 365."""
    closes = closes.copy()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize("UTC")
    idx = pd.date_range(START, END, freq="D", tz="UTC")
    full = closes.reindex(idx.union(closes.index))
    listed = full.notna().cummax()
    signal = full.ffill().where(listed).reindex(idx)
    nxt = full.shift(-1).bfill()  # value at the first row after d that has a close
    nxt = nxt.where(listed.shift(-1, fill_value=True)).reindex(idx)
    nxt = nxt.where(signal.notna())
    funding = bitget_rate_drag(funding_pct / 100.0, idx, closes.columns)
    slip = pd.DataFrame(np.tile(slippage_bps.clip(lower=MIN_SLIP_BPS).reindex(closes.columns).to_numpy(), (len(idx), 1)),
                        index=idx, columns=closes.columns)
    return {"signal_close": signal, "exec_close": nxt, "funding": funding, "slippage_bps": slip}


def scaled(panels: dict, mult: float) -> dict:
    out = dict(panels)
    out["funding"] = panels["funding"] * mult
    out["slippage_bps"] = panels["slippage_bps"] * mult
    return out


def cap_gross(w: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    g = w.abs().sum(axis=1)
    return w.mul((max_gross / g).clip(upper=1.0).fillna(1.0), axis=0)


def run(panels: dict, targets: pd.DataFrame, band: float, weekly: bool = True, fee: float = FEE):
    """Engine run; returns (daily with 'net' = conservative funding, 'net_symmetric', detail)."""
    idx = panels["exec_close"].index
    reb = idx[idx.weekday == DECISION_WEEKDAY] if weekly else None
    daily, det = backtest_weights(panels, targets, fee=fee, band=band, rebalance_on=reb,
                                  rebalance_days=1, detail=True)
    paid = det["funding"].clip(upper=0.0).sum(axis=1)  # never credit the receiving side
    daily = daily.assign(net_symmetric=daily["net"], funding_symmetric=daily["funding"])
    daily["funding"] = paid
    daily["net"] = daily["gross"] + paid - daily["cost"]
    return daily, det


def stats(daily: pd.DataFrame, col: str = "net") -> dict:
    d = daily if col == "net" else daily.assign(net=daily[col], funding=daily["funding_symmetric"])
    return {"is": summarize(d, START, SPLIT), "oos": summarize(d, SPLIT, END)}


def active(net: pd.Series, daily: pd.DataFrame, start: str, end: str) -> pd.Series:
    m = (daily.index >= pd.Timestamp(start, tz="UTC")) & (daily.index < pd.Timestamp(end, tz="UTC"))
    on = daily["gross_exposure"].shift(1, fill_value=0.0).gt(0) | daily["turnover"].gt(0)
    return net[m & on]


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(365.0)) if len(x) > 30 and x.std() > 0 else float("nan")


def max_dd(x: pd.Series) -> float:
    eq = (1.0 + x).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def bootstrap_p_sharpe_le_0(net: pd.Series, block: int = 20, reps: int = 2000, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    x, n = net.to_numpy(), len(net)
    out = []
    for _ in range(reps):
        starts = rng.integers(0, n, size=n // block + 1)
        s = x[(starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n]
        out.append(s.mean() / s.std())
    return float((np.array(out) <= 0).mean())


def by_year(daily: pd.DataFrame) -> dict:
    out = {}
    for y, g in daily.groupby(daily.index.year):
        x = active(g["net"], g, f"{y}-01-01", f"{y + 1}-01-01")
        if len(x) > 30:
            out[int(y)] = round(sharpe(x), 2)
    return out


def side_split(det: dict, start: str) -> dict:
    m = det["weight"].index >= pd.Timestamp(start, tz="UTC")
    w = det["weight"][m]
    pnl = det["gross"][m] + det["funding"][m].clip(upper=0.0)
    return {"long_ann": float(pnl.where(w > 0, 0.0).sum(axis=1).mean() * 365),
            "short_ann": float(pnl.where(w < 0, 0.0).sum(axis=1).mean() * 365)}


def h1_targets(panels: dict, cfg: TsmomConfig) -> pd.DataFrame:
    return cap_gross(tsmom_targets(panels["signal_close"], cfg), H1_MAX_GROSS)


def study_h1(panels: dict, classes: pd.Series, crypto: pd.Series, log) -> dict:
    targets = h1_targets(panels, H1)
    daily, det = run(panels, targets, H1.band)
    s = stats(daily)
    oos = active(daily["net"], daily, SPLIT, END)
    contrib = (det["gross"] + det["funding"].clip(upper=0.0) - det["cost"])
    contrib = contrib[contrib.index >= pd.Timestamp(SPLIT, tz="UTC")].sum()
    by_class = contrib.groupby(classes).sum().sort_values(ascending=False)
    best = by_class.index[0]
    keep = (classes.reindex(targets.columns) != best).astype(float)
    ex_best, _ = run(panels, targets.mul(keep, axis=1), H1.band)
    ex_best_sharpe = sharpe(active(ex_best["net"], ex_best, SPLIT, END))
    p_boot = bootstrap_p_sharpe_le_0(oos)
    results = {
        "oos_sharpe": s["oos"]["sharpe"], "is_sharpe": s["is"]["sharpe"], "oos_max_dd": s["oos"]["max_dd"],
        "best_class": best, "oos_sharpe_ex_best_class": ex_best_sharpe, "bootstrap_p": p_boot,
    }
    checks = {
        "oos_sharpe": results["oos_sharpe"] >= H1_CRITERIA["oos_sharpe_min"],
        "is_sharpe": results["is_sharpe"] >= H1_CRITERIA["is_sharpe_min"],
        "oos_max_dd": results["oos_max_dd"] >= H1_CRITERIA["oos_max_dd_min"],
        "ex_best_class": ex_best_sharpe >= H1_CRITERIA["oos_sharpe_ex_best_class_min"],
        "bootstrap": p_boot < H1_CRITERIA["bootstrap_p_max"],
    }
    log(f"H1 dev {s['is']}\nH1 oos {s['oos']}\nH1 checks {checks}")

    # Combination with the crypto momentum primary, equal risk scaled on its first years.
    tf = daily["net"]
    a, b = pd.Timestamp(COMBO["scale_start"], tz="UTC"), pd.Timestamp(COMBO["scale_end"], tz="UTC")
    common = crypto.index.intersection(tf.index)
    cr, tf = crypto.reindex(common), tf.reindex(common)
    k = float(cr[(common >= a) & (common < b)].std() / tf[(common >= a) & (common < b)].std())
    combo = 0.5 * cr + 0.5 * k * tf
    ev = common >= pd.Timestamp(COMBO["eval_start"], tz="UTC")
    comb = {"scale": k, "corr": float(cr[ev].corr(tf[ev])),
            "crypto_sharpe": sharpe(cr[ev]), "combined_sharpe": sharpe(combo[ev]),
            "crypto_max_dd": max_dd(cr[ev]), "combined_max_dd": max_dd(combo[ev]),
            "implied_gross_mean": float(daily["gross_exposure"][daily.index >= a].mean() * k * 0.5)}
    comb["useful"] = bool(comb["combined_sharpe"] >= comb["crypto_sharpe"] + COMBO["sharpe_gain_min"]
                          and comb["combined_max_dd"] > comb["crypto_max_dd"])

    sym = stats(daily, "net_symmetric")
    expo = daily["gross_exposure"][daily.index >= pd.Timestamp(SPLIT, tz="UTC")]
    report = {
        "config": H1.to_dict(), "max_gross": H1_MAX_GROSS, "criteria": H1_CRITERIA, "results": results,
        "checks": checks, "passed": all(checks.values()), "summary": s, "symmetric_funding": sym,
        "oos_contribution_by_class": by_class.round(4).to_dict(),
        "oos_contribution_by_asset": contrib.sort_values(ascending=False).round(4).to_dict(),
        "sharpe_by_year": by_year(daily), "sides_oos": side_split(det, SPLIT),
        "gross_exposure_oos": {"mean": float(expo.mean()), "p95": float(expo.quantile(0.95))},
        "combination": comb,
    }
    return report, daily


def grid_h1(panels: dict) -> pd.DataFrame:
    rows = []
    for lb, weekly, mult in itertools.product(GRID["lookbacks"], GRID["weekly"], GRID["cost_mult"]):
        cfg = replace(H1, lookbacks=lb)
        d, _ = run(scaled(panels, mult), h1_targets(panels, cfg), cfg.band, weekly=weekly, fee=FEE * mult)
        s = stats(d)
        rows.append({"lookbacks": "/".join(map(str, lb)), "weekly": weekly, "cost_mult": mult,
                     "is_sharpe": s["is"].get("sharpe"), "oos_sharpe": s["oos"].get("sharpe"),
                     "oos_max_dd": s["oos"].get("max_dd"), "oos_cagr": s["oos"].get("cagr")})
    return pd.DataFrame(rows)


def study_h2(panels: dict, cfg: MomentumConfig) -> dict:
    targets = target_weights(panels["signal_close"], cfg)
    daily, det = run(panels, targets, cfg.band)
    s = stats(daily)
    contrib = (det["gross"] + det["funding"].clip(upper=0.0) - det["cost"])
    contrib = contrib[contrib.index >= pd.Timestamp(SPLIT, tz="UTC")].sum().sort_values(ascending=False)
    top3 = list(contrib.index[:3])
    rest = {k: v.drop(columns=top3) for k, v in panels.items()}
    d3, _ = run(rest, target_weights(rest["signal_close"], cfg), cfg.band)
    ex3 = sharpe(active(d3["net"], d3, SPLIT, END))
    return {"config": cfg.to_dict(), "summary": s, "symmetric_funding": stats(daily, "net_symmetric"),
            "top3_oos": top3, "oos_sharpe_ex_top3": ex3, "sharpe_by_year": by_year(daily),
            "sides_oos": side_split(det, SPLIT), "oos_contribution_by_stock": contrib.round(4).to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache/tradfi")
    ap.add_argument("--out", default="artifacts/research/tradfi_v1")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = print

    proxies = {k: v[0] for k, v in MACRO.items()}
    if not args.no_fetch:
        update_yahoo(list(proxies.values()) + STOCKS, args.cache_dir)
    closes = load_closes(list(proxies.values()), args.cache_dir).rename(columns={v: k for k, v in proxies.items()})
    meta = pd.DataFrame(MACRO, index=["proxy", "class", "impact_bp", "funding_pct"]).T
    panels = build_panels(closes, meta["impact_bp"].astype(float), meta["funding_pct"].astype(float))

    crypto = pd.read_csv(CRYPTO_DAILY, index_col=0, parse_dates=True)["momentum"]
    h1, h1_daily = study_h1(panels, meta["class"], crypto, log)
    grid = grid_h1(panels)
    h1["grid_frac_oos_sharpe_ge_0_3"] = float((grid["oos_sharpe"] >= 0.3).mean())
    grid.to_csv(out / "grid_h1.csv", index=False)

    sc = pd.read_csv(STOCK_COSTS).set_index("sym")
    stock_closes = load_closes(STOCKS, args.cache_dir)
    sp = build_panels(stock_closes, sc["impact10k_bp"].reindex(STOCKS), (sc["funding_ann"] * 100).reindex(STOCKS))
    h2 = study_h2(sp, H2)
    r = h2["summary"]
    h2["checks"] = {"oos_sharpe": r["oos"]["sharpe"] >= H2_CRITERIA["oos_sharpe_min"],
                    "is_sharpe": r["is"]["sharpe"] >= H2_CRITERIA["is_sharpe_min"],
                    "oos_max_dd": r["oos"]["max_dd"] >= H2_CRITERIA["oos_max_dd_min"],
                    "ex_top3": h2["oos_sharpe_ex_top3"] >= H2_CRITERIA["oos_sharpe_ex_top3_min"]}
    h2["passed"] = all(h2["checks"].values())
    h2_expl = study_h2(sp, H2_EXPLORATORY)
    log(f"H2 {h2['summary']}\nH2 checks {h2['checks']}\nH2 exploratory {h2_expl['summary']}")

    h1_daily[["net", "net_symmetric", "gross_exposure"]].to_csv(out / "daily_h1.csv")
    report = {"h1": h1, "h2": h2, "h2_exploratory_crypto_config": h2_expl,
              "periods": {"start": START, "split": SPLIT, "end": END}}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
