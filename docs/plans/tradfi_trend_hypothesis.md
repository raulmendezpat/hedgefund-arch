# Registered hypothesis: trend following and stock momentum on Bitget's traditional-asset perpetuals

Written on 2026-10-31, **before running any strategy backtest on these assets**. Only cost and
tracking inputs were measured beforehand (below); no strategy return had been computed. The primary
configuration and the success criteria are not changed after seeing results. The grids measure
robustness; they are not used to choose parameters.

## Why

Bitget lists 404 USDT-M perpetuals on traditional assets (`isRwa: YES`): US stocks, equity and bond
ETFs, stock indices, metals, energy and FX. They use the same account, API, fees (0.06% taker,
0.02% maker), 5 USDT minimum order and 24/7 trading as the crypto perpetuals, so the existing
pipeline (signal → target weights → `hf_research.engine` → live runner) can trade them unchanged
from a separate subaccount.

Trend following across asset classes (time-series momentum) is the best-documented strategy for
this kind of universe: positive in each decade for over a century in futures (Hurst, Ooi &
Pedersen, 2017, "A Century of Evidence on Trend-Following Investing"; Moskowitz, Ooi & Pedersen,
2012). It also tends to have low correlation with the crypto momentum book.

## Data and what was measured before registering (2026-10-31)

**History:** Bitget's history for these perpetuals starts in 2026-08, so the backtest uses each
underlying's daily close from Yahoo Finance (`hf_research.tradfi`), split-adjusted and **without
dividends** (a perpetual holder does not receive them).

**Tracking** (2026-08-03 to 2026-10-30, weekly returns of the Bitget perpetual against the proxy;
raw measurements in `artifacts/research/tradfi_v1/measured_20261031/`):
- Equity ETFs, bonds, metals: correlation 0.99–1.00; the perpetual's price follows the ETF or spot
  price.
- Oil and copper: the perpetual follows the front futures price, and its funding plays the role of
  the futures roll. Perpetual price + funding is within 4–10%/yr of the rolled ETFs (USO, BNO,
  CPER); natural gas is 32%/yr worse than UNG.
- Excluded for poor tracking (< 0.85): USDJPY (0.57), JP225 (0.87 vs EWJ 0.99 is kept instead).

**Costs measured on Bitget's live order books** (average of buy and sell, one way, for a 10,000
USDT market order) and **funding** (annualised mean of all available history, longs pay when
positive):

| Perpetual | Proxy | Impact 10k (bp) | Funding (%/yr) |
|---|---|---|---|
| SPY | SPY | 1.3 | 3.4 |
| QQQ | QQQ | 1.4 | 6.3 |
| IWM | IWM | 11.4 | 0.1 |
| EWJ | EWJ | 13.5 | 0.0 |
| EWY | EWY | 3.6 | 3.6 |
| EWT | EWT | 18.4 | −0.7 |
| HSI | ^HSI | 2.3 | 4.3 |
| TLT | TLT | 6.1 | 2.6 |
| XAU | GLD | 0.0 | 12.0 |
| XAG | SLV | 1.6 | 12.4 |
| XPT | PPLT | 4.2 | 19.3 |
| XPD | PALL | 18.0 | 24.8 |
| CL | USO | 1.0 | 3.7 † |
| BZ | BNO | 1.1 | 9.7 † |
| NATGAS | UNG | 5.5 | 32.2 † |
| COPPER | CPER | 8.8 | 5.1 † |
| EURUSD | EURUSD=X | 3.2 | −0.7 |
| GBPUSD | GBPUSD=X | 4.3 | −6.8 |
| USDBRL | BRL=X | 7.0 | 2.6 |

† The rolled ETF already contains the roll, so the funding used is the measured residual: the
perpetual's price return plus funding, minus the ETF's return, over the same days.

Metals stand out: longs pay a 12–25%/yr premium over spot.

## Universe M (macro, primary): the 19 perpetuals above

Rule: one perpetual per market, 10k impact ≤ 20 bp, tracking ≥ 0.85, and a proxy with daily
history. Excluded by rule: duplicates (VOO, SP500, NDX100, DIA), EWZ (22 bp), EWH (50 bp), INDA
(36 bp, tracking 0.86), sector ETFs (≥ 15 bp), USDJPY and JP225 (tracking).

An asset enters when its proxy has 365 days of history (PPLT/PALL 2011, BNO 2011, CPER 2012).

## Cost model

- Fee 0.06% per side plus the measured 10k impact per asset as slippage (floor 2 bp).
- **Decision model (conservative):** funding is charged when the position is on the paying side and
  **never credited** on the receiving side. The current funding level reflects today's crowding
  (for example, long gold), which need not hold historically.
- Reported alongside, not deciding: the symmetric model (the receiving side earns the funding).
- Idle cash earns nothing.

## H1 (primary): multi-asset time-series momentum on universe M

Same code as the BTC/ETH TSMOM (`hf_research.tsmom.tsmom_targets`), with the standard
managed-futures parameters from the literature:
- **Signal:** mean of sign(r_30), sign(r_90), sign(r_365), with r_L the proxy's return over L
  calendar days (~1, 3 and 12 months).
- **Size:** w_i = s_i × 0.03 / σ_i, |w_i| ≤ 0.5, with σ_i the annualised volatility over the last
  60 calendar days. With 19 assets this aims at roughly 10–15% portfolio volatility. If the gross
  exposure exceeds 3.0, all weights are scaled down to 3.0.
- **Schedule:** weekly. Decision at Friday's close, fill at the next trading day's close (normally
  Monday). Band 0.01.
- **Periods:** development 2008-01-01 to 2017-12-31; out of sample 2018-01-01 to 2026-10-30.

### Success criteria (out of sample, conservative cost model, all required)

| # | Criterion |
|---|---|
| 1 | Out-of-sample Sharpe ≥ 0.5 |
| 2 | Development Sharpe ≥ 0.3 |
| 3 | Out-of-sample worst drawdown ≥ −20% |
| 4 | Out-of-sample Sharpe with the best-contributing asset class removed (equities, bonds, metals, energy, FX) ≥ 0.25 |
| 5 | 20-day block bootstrap: P(out-of-sample Sharpe ≤ 0) < 10% |

**Combination with the crypto momentum primary** (its daily net series from
`artifacts/research/tsmom_btc_eth_v1/daily_net.csv`, 2021-03-01 to 2026-10-19): equal risk, with
the scale computed on 2021-03-01 to 2023-12-31. Useful if, over 2024-01-01 to 2026-10-19, the
combined Sharpe ≥ crypto momentum + 0.15 **and** the worst drawdown is better. The daily
correlation is reported.

**Also reported, not deciding:** symmetric funding; Sharpe by year and by asset class; long and
short contributions; gross exposure (mean and p95); turnover and cost.

### Sensitivity (robustness only)

- Lookbacks: (30), (90), (365), (30, 90, 365), (20, 60, 120).
- Rebalance: weekly and daily.
- Costs: as measured and ×2 (fee, slippage and funding).

The fraction of the 20 combinations with out-of-sample Sharpe ≥ 0.3 is reported.

## H2 (secondary): cross-sectional 12–1 momentum on single stocks

**Universe:** Bitget perpetuals on US-listed single stocks with 10k impact ≤ 10 bp (measured
2026-10-31) and a Yahoo history starting before 2012-01-01. Funding: the measured mean of the last
~33 days per stock (noisy; for example INTC 15%/yr, GOOGL 14%, TSLA 10%, AAPL 0%).

By that rule (29 stocks): INTC MU MSTR GOOGL NVDA AMZN TSLA AMD MRVL AAPL MSFT AVGO WDC COHR ORCL GLW
TSM QCOM ASML NKE FTNT BP SBUX AKAM PBR MDT DHR EBAY HD. Each enters when it has 365 days of
history.

**Known bias, stated in advance:** the list is today's survivors that Bitget chose to list, mostly
because they are popular now. That flatters the long side of any backtest. **A pass does not
authorise anything beyond a paper test; a failure is conclusive.**

**Configuration:** `hf_research.xs_momentum.target_weights` with lookback 365 days and skip 30
days (the standard 12–1 month stock momentum), 20% per side, equal weights 0.5 / −0.5, at least 8
stocks with a score, weekly with the H1 schedule, band 0.02. Same cost model and periods as H1.

**Criteria (all required):** out-of-sample Sharpe ≥ 0.5, development Sharpe ≥ 0.3, out-of-sample
worst drawdown ≥ −30%, out-of-sample Sharpe without the 3 best-contributing stocks ≥ 0.25.

**Exploratory, decides nothing:** the crypto primary configuration as it is (30-day return, 1-day
skip). Over one month, stocks are known for short-term reversal, so this is expected to fail.

## What a pass leads to (design, not built yet)

Only if H1 passes. If only H2 passes, nothing goes live: H2 can at most become a paper test.
1. A paper runner with the configuration frozen here, signal from the proxies' closes and fills
   simulated against Bitget's order book, like the other papers.
2. A live runner on **its own Bitget subaccount**, reusing `hf_research.live_xs` exactly as
   `scripts/live_tsmom.py` does: its own credentials file, refusal of the other runners' keys,
   the same kill switch, sanity limits and dry-run by default.
3. The same implementation gate as the other papers before any capital.

## Known limitations

- **Short Bitget history:** the funding and impact inputs come from 1–3 months of data. Funding on
  these perpetuals may change a lot as the market matures. That is why the decision model never
  credits funding, and why costs ×2 are tested.
- **Proxy timing:** proxies close at different times (Hong Kong vs New York vs FX). Weekly
  decisions on daily closes with a one-day fill lag keep this small, but it is not exact.
- **Weekends and holidays:** the perpetuals trade when the underlying market is closed. The
  backtest carries the last close (no weekend moves), while live positions bear weekend gaps.
- **Rates:** constant funding ignores that interest rates were near zero from 2009 to 2021, when
  the carry part of equity funding would have been lower.

## Result (2026-10-31): both hypotheses rejected

Run: `scripts/run_tradfi_study.py` → `artifacts/research/tradfi_v1/`.

**Data correction (not a parameter change).** Yahoo's history has bad prints: one-day spikes that
revert the next day, for example EURUSD on the 8th of five months of 2008 (+6% to +17%) and CPER
−33% then +50% (2014-12-04). The first run used the raw data. `tradfi.drop_bad_ticks` now removes
isolated reverting spikes in FX series (> 6σ and > 3%, calm days before) and absurd ones elsewhere
(> 12σ and > 20%). It removes 5 EURUSD and 2 CPER closes, and keeps real reverting shocks (March
2020, TSLA 2012-01-13, natural gas 2018-11-14). All of them fall in the development period; the
out-of-sample results are identical before and after.

### H1: multi-asset trend (rejected on every criterion)

| Criterion | Threshold | Result | Met? |
|---|---|---|---|
| Out-of-sample Sharpe | ≥ 0.5 | −0.27 | no |
| Development Sharpe | ≥ 0.3 | −0.48 | no |
| Out-of-sample worst drawdown | ≥ −20% | −61% | no |
| Sharpe without the best class (equities) | ≥ 0.25 | −0.48 | no |
| Bootstrap P(Sharpe ≤ 0) | < 10% | 79% | no |

- **The signal works; Bitget's costs do not let it pay.** Out of sample: gross +8.3%/yr (gross
  Sharpe 0.46, in line with published trend indices over the same years), funding −7.3%/yr, fees
  and slippage −5.8%/yr.
  - Turnover is 46–49 times capital a year, at a mean gross exposure of 2.2. At 0.06% fee plus
    ~5 bp of slippage, that is ~5.5%/yr. A futures trend fund pays well under 1%/yr for the same
    trades.
  - 2022, the best trend year in decades: the book was correctly short bonds, the euro, the pound
    and Asian equities and long oil, and made +10.8% gross; funding (−4.7%) and costs (−4.5%) took
    85% of it.
- **Symmetric funding** (crediting the receiving side): out-of-sample Sharpe 0.04, development
  −0.10. Still fails.
- **By class, out of sample (cumulative):** equities +6.6%, energy −0.6%, bonds −1.4%, metals
  −19.7% (longs pay 12–25%/yr), FX −27.8%.
- **Grid:** none of the 20 combinations reaches an out-of-sample Sharpe of 0.3; the best is the
  365-day signal alone, weekly (−0.03).
- **Combination with crypto momentum:** correlation −0.04, but the combined Sharpe falls from 1.03
  to 0.76. Not useful.

**Optimistic bound (exploratory, `exploratory_upper_bound.txt`):** even with maker fees (0.02%), no
slippage and symmetric funding, the primary has an out-of-sample Sharpe of 0.31 (drawdown −41%).
With no costs at all, the best variant (365 days) reaches 0.59 out of sample, 0.30 in development,
with a −34% drawdown: well below crypto momentum even before paying anything.

### H2: single-stock 12–1 momentum (rejected, despite the bias in its favour)

| Criterion | Threshold | Result | Met? |
|---|---|---|---|
| Out-of-sample Sharpe | ≥ 0.5 | −0.24 | no |
| Development Sharpe | ≥ 0.3 | −0.26 | no |
| Out-of-sample worst drawdown | ≥ −30% | −52% | no |
| Sharpe without the 3 best stocks (MSTR, NVDA, AMD) | ≥ 0.25 | −0.66 | no |

- The longs earned +14.9%/yr out of sample, but the shorts lost −17.7%/yr: in a 29-stock universe of
  today's winners, the shorts are also winners.
- Funding cost 6.8%/yr out of sample, more than the gross return (+4.0%/yr). Symmetric funding:
  Sharpe −0.01.
- Since the survivor bias flatters this test, the failure is conclusive.
- **Exploratory (crypto configuration, 30-day return):** Sharpe −0.38 in development and −0.30
  out of sample, as expected from short-term reversal in stocks.

**Conclusion:** on Bitget, trading traditional assets through perpetuals costs too much for these
strategies. The fee is the same as for crypto, but these assets move 3–5 times less, so it is a far
larger share of their return. Funding premia make it worse, mostly for metals and popular stocks.
Nothing goes to paper, and no runner is built. Worth revisiting only if Bitget's fees on these
perpetuals or their funding premia fall a lot. In that case, re-measure the inputs and register a
new test.
