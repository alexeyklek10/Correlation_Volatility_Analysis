# Securities Analysis: Correlation & Volatility

Two Jupyter notebooks I use to analyse correlation structure and volatility
regimes across a multi-asset ETF book. They fetch from Yahoo Finance, cache
locally, and produce a fixed dashboard plus a console scorecard. Everything
runs from a single Colab cell or a local Python 3.10+ environment.

## Sample output

`images/correlation/01_executive_dashboard.png` is the headline page for the
correlation side: an 8-panel layout showing portfolio ρ and β to the SPY
anchor, tail behaviour, the drawdown profile, and the risk decomposition.

`images/volatility/GLD_dashboard1_core.png` is the representative per-asset
volatility dashboard. It overlays seven estimators (Realized, Parkinson,
Garman-Klass, Rogers-Satchell, Yang-Zhang, EWMA, GARCH(1,1)) and labels the
current regime.

## Methodology

- **Rank correlation as the backbone.** Spearman ρ_S (primary) and Kendall τ
  (secondary). Pearson and the RiskMetrics-style EWMA estimator were removed
  in the rank-correlation rework: multi-asset returns are fat-tailed and the
  pairwise relationship between asset classes is rarely linear, so a linear
  estimator is the wrong tool. Beta to SPY and the variance decomposition
  remain covariance-based by definition (beta is an OLS regression slope;
  portfolio variance is `wᵀΣw`, a covariance identity) — these are explicitly
  not "rank-ified".
- **Tail-regime correlation at three percentiles.** Spearman ρ_S restricted
  to the worst 5%, 10%, and 15% of anchor (SPY) days, so the "diversification
  breaks in tails" hypothesis can be quantified rather than asserted. Rank
  correlation is more appropriate for tail subsets anyway, where a handful
  of extreme observations would otherwise dominate Pearson.
- **Variance decomposition.** Co-movement share (sum of cross-asset
  covariance contributions) vs diagonal share (sum of each asset's own
  variance contribution). This is distinct from CAPM-style market-factor
  decomposition, which is not computed here.
- **PCA + ENB diversification scorecard.** Eigendecomposition of the
  **Spearman correlation matrix** (not standardized returns), so the factor
  structure is rank-consistent with the rest of the suite. Reports variance
  share of PC1/PC2, the number of PCs needed to clear 80% / 90% cumulative
  variance, and the effective number of independent bets via the entropy-
  based ENB metric.
- **Hierarchical clustering with Mantegna's distance**
  `d_ij = sqrt(0.5 * (1 - rho_ij))` (Mantegna 1999), which is a true
  ultrametric on a correlation matrix. Ward linkage.
- **OHLC volatility estimators.** Realized (close-to-close), Parkinson
  (high-low), Garman-Klass (OHLC), Rogers-Satchell (drift-robust OHLC),
  Yang-Zhang (Garman-Klass plus overnight). On range-trading or drift-heavy
  series the range-based estimators have roughly 5x the efficiency of plain
  close-to-close.
- **GARCH(1,1) MLE** fitted per asset via `arch.arch_model`, reporting α, β,
  ω, persistence (α + β), and half-life log(0.5) / log(α + β).
- **Bias-corrected Hurst exponent.** Anis-Lloyd (1976) closed-form
  correction subtracted from the empirical R/S slope, so the published
  H ≈ 0.5 for iid Gaussian noise rather than the upward-biased ~0.62 a naive
  R/S regression returns.
- **Lo-MacKinlay variance ratio.** Random-walk null with the
  overlapping-returns asymptotic SE `sqrt(2 * (2k-1) * (k-1) / (3*k*T))` at
  lags 2, 5, 10, 20.
- **Multiple-testing correction.** Pairwise Spearman significance reported
  with Benjamini-Hochberg FDR at α = 0.05 (p-values from `scipy.stats.spearmanr`),
  not just uncorrected p-values.

## How to run

### Local

```bash
git clone https://github.com/alexeyklek10/Security_Analysis.git
cd Security_Analysis
python -m venv .venv
.venv/Scripts/activate              # Windows
# or: source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
jupyter notebook notebooks/
```

Open either notebook and run all cells. The first run populates `data/cache/`
with parquet files (~10 MB); subsequent runs are seconds.

### Colab

Open via the badge below. The first cell of the volatility notebook installs
`arch` on first run; everything else is already in Colab's default image.

[![Open Correlation_Analysis.ipynb in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexeyklek10/Security_Analysis/blob/main/notebooks/Correlation_Analysis.ipynb)
[![Open Volatility_Analysis.ipynb in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexeyklek10/Security_Analysis/blob/main/notebooks/Volatility_Analysis.ipynb)

## Interpretation guide

| Metric | What it means | Typical / good values |
|---|---|---|
| Portfolio ρ_S to SPY | Spearman rank correlation of the weighted portfolio to SPY — robust to outliers and non-linear monotone moves | Multi-asset book: 0.3 to 0.7. Single-factor: 0.85+ |
| Portfolio β to SPY | OLS regression slope vs SPY (linear by definition, not rank-ified) | Defensive: 0.3 to 0.6; balanced: 0.6 to 0.9; aggressive: > 0.9 |
| Avg pairwise correlation | Mean of the upper-triangle **Spearman ρ_S** matrix | < 0.30: well-diversified; 0.30 to 0.50: typical; > 0.50: redundant |
| Diversification ratio | Weighted avg vol ÷ portfolio vol | > 1.5: strong; 1.2 to 1.5: moderate; < 1.2: weak |
| Co-movement share | % of portfolio variance from cross-asset covariances | Lower means more diversified |
| Diagonal share | % of portfolio variance from per-asset variances | Higher means more diversified |
| ENB (effective N bets) | Entropy-equivalent independent-bet count | N_assets / 2 is the baseline; > 0.7 × N is excellent |
| GARCH persistence (α + β) | How long shocks live in the vol process | 0.94 to 0.98 normal; > 0.99 near-unit-root |
| GARCH half-life | Days for a vol shock to decay 50% | 15 to 30 days is typical for equity ETFs |
| Hurst exponent (bias-corrected) | < 0.5 mean-reverting, ≈ 0.5 random walk, > 0.5 trending | 0.42 to 0.58 is "indistinguishable from random walk" |
| Yang-Zhang vol | Most efficient OHLC vol estimator on gappy data | Closer to GARCH than to close-to-close on intraday-active assets |

## Limitations

A few caveats worth flagging before reading too much into any single number.

- **Single-number kurtosis is a noisy diagnostic for series with
  concentrated extreme events.** The cross-asset comparison table flags any
  excess kurtosis above 20 as a candidate for the per-asset distribution
  plot rather than for taking the kurtosis number at face value. The
  volatility notebook also includes an opt-in top-5 |log_return| days
  diagnostic for tickers known to have vendor data artifacts; it is
  inactive on the default standard-ETF universe.
- **GARCH(1,1) assumes a single volatility regime.** It cannot separately
  model a calm process and a stressed process. Persistence estimates near
  1.0 should be read as "the volatility process may be non-stationary on
  this sample" rather than as a number to forecast with.
- **Rolling-window choices are convention-dependent.** 21-day vs 63-day vs
  252-day realized vol will give different regime classifications. The
  notebook reports all three but does not claim a single "right" window.
- **No transaction costs, slippage, or liquidity premia.** Correlation and
  volatility estimates are pre-cost; any backtested portfolio impact
  derivable from these numbers is gross-of-cost.
- **Rank correlation is not Pearson.** Spearman ρ_S and Kendall τ measure
  monotone co-movement, not linear co-movement. For two assets that move
  together in rank but with very different magnitudes, ρ_S can be high while
  a Pearson correlation would be lower. That is the desired property here
  — magnitude is captured separately by per-asset vol and by beta to SPY —
  but it means correlations reported here are NOT directly substitutable
  into closed-form formulas that assume Pearson (e.g. textbook Markowitz
  variance from `wᵀΣw` reconstituted as `vol·corr·vol`). The variance
  decomposition in this notebook uses the actual sample covariance for that
  reason; it does not factor through the Spearman matrix.
- **Yahoo data quality is uneven.** The volatility notebook applies a
  general filter for vol=0 price-change rows (vendor stub artifacts) before
  computing returns. A defensive named-row mask in `fetch_ohlc_data` covers
  one historical Yahoo flash-print on BTAL (2015-04-29) that the general
  filter cannot catch because volume on that row is non-zero. The mask is
  inactive on the default universe and stays in the code as a worked
  example of how to address a similar issue if it surfaces elsewhere.

## Development

```bash
# One-time: enable the nbstripout filter so notebook outputs don't end up
# in diffs.
.venv/Scripts/nbstripout --install
.venv/Scripts/pytest tests/
```

The notebooks stay Colab-pasteable, so the cache helper and `apply_theme()`
are duplicated byte-for-byte between them rather than imported from a shared
module. If you modify either, copy the change to the other.

## License

MIT. See [`LICENSE`](LICENSE).
