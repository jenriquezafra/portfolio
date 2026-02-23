# ML - Portfolio Optimization (XGBoost + Mean-Variance)

A minimal quantitative portfolio construction framework combining:

- Conditional return estimation via **Machine Learning (XGBoost)**
- Classical **Mean–Variance Optimization**
- Strict **walk-forward backtesting**
- Explicit control of **risk, turnover, and transaction costs**

Project changelog: `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/CHANGELOG.md`

---

# 1. Theoretical Framework

## 1.1 Notation

- Universe of **N** equities
- At time $t$, future returns over horizon $H$:

$$
r_{t+H} \in \mathbb{R}^N
$$

- Portfolio weights:

$$
w_t \in \mathbb{R}^N
$$

with constraints:

$$
\mathbf{1}^T w_t = 1
$$
$$
w_{i,t} \ge 0
$$
$$
w_{i,t} \le w_{\max}
$$

---

# 2. Classical Portfolio Theory

The foundation is the Mean–Variance framework of Harry Markowitz.

The investor solves:

$$
\max_{w_t} \quad w_t^T \mu_t - \frac{\lambda}{2} w_t^T \Sigma_t w_t
$$

where:

- $\mu_t = E[r_{t+H} \mid \mathcal{F}_t]$
- $\Sigma_t = Cov(r_{t+H} \mid \mathcal{F}_t)$
- $\lambda > 0$ is the risk aversion parameter

This corresponds to maximizing quadratic utility:

$$
U(w_t) = E[w_t^T r_{t+H}] - \frac{\lambda}{2} Var(w_t^T r_{t+H})
$$

---

# 3. The Core Problem: Estimation of Expected Returns

In classical implementations:

$$
\mu_t = \text{historical mean}
$$

However:

- Expected returns are noisy
- Estimation error dominates optimization
- Small errors in $\mu_t$ produce extreme weight instability

Thus, estimating conditional expected returns is the central challenge.

---

# 4. Machine Learning as Conditional Expectation Estimator

We model:

$$
\hat{\mu}_{i,t} = E[r_{i,t+H} \mid X_{i,t}]
$$

where:

- $X_{i,t}$ is a feature vector
- $\hat{\mu}_{i,t}$ is the predicted conditional return

Formally:

$$
\hat{\mu}_{i,t} = f_\theta(X_{i,t})
$$

with:

$$
f_\theta = \text{XGBoost}
$$

---

## 4.1 Interpretation

We assume that:

$$
E[r_{t+H} \mid X_t] \neq 0
$$

i.e., the market exhibits predictable structure (momentum, volatility clustering, etc.).

Machine learning approximates the conditional expectation operator:

$$
f_\theta \approx E[r_{t+H} \mid X_t]
$$

---

# 5. XGBoost Model

XGBoost is a gradient-boosted ensemble of decision trees:

$$
f_\theta(x) = \sum_{m=1}^M \gamma_m h_m(x)
$$

where:

- $h_m$ are regression trees
- $\gamma_m$ are weights
- $M$ is number of boosting rounds

Training objective:

$$
\min_\theta \sum_{t,i} (r_{i,t+H} - f_\theta(X_{i,t}))^2 + \Omega(\theta)
$$

where $\Omega(\theta)$ is regularization.

---

# 6. Portfolio Construction with ML Predictions

Once predictions are obtained:

$$
\hat{\mu}_t = (\hat{\mu}_{1,t}, \dots, \hat{\mu}_{N,t})
$$

we solve:

$$
\max_{w_t} \quad w_t^T \hat{\mu}_t - \frac{\lambda}{2} w_t^T \Sigma_t w_t
$$

Optionally including turnover control:

$$
\max_{w_t} \quad w_t^T \hat{\mu}_t - \frac{\lambda}{2} w_t^T \Sigma_t w_t - \eta \| w_t - w_{t-1} \|^2
$$

where:

- $\eta > 0$ penalizes excessive rebalancing
- This reduces transaction costs and instability

---

# 7. Risk Estimation

Covariance matrix estimated via:

- Rolling sample covariance
- EWMA
- Shrinkage estimators

$$
\Sigma_t = \widehat{Cov}(r_{t-L:t})
$$

with lookback window $L$.

---

# 8. Backtesting Framework

Walk-forward procedure:

For each rebalancing date $t_k$:

1. Train model on:
$$[t_k - T_{\text{train}}, \, t_k]$$

2. Predict: 
$$\hat{\mu}_{t_k}$$

3. Estimate covariance:
$$\Sigma_{t_k}$$

4. Solve optimization problem

5. Hold portfolio until next rebalance

---

# 9. Performance Metrics

- Portfolio return:

$$R_{p,t} = w_t^T r_{t}$$

- Sharpe ratio:

$$S = \frac{E[R_p]}{\sqrt{Var(R_p)}}$$

- Maximum drawdown

- Turnover:

$$TO_t = \sum_i |w_{i,t} - w_{i,t-1}|$$

- Information Coefficient (IC):

$$IC_t = Corr(\hat{\mu}_{t}, r_{t+H})$$

---

# 10. Conceptual Summary

The system implements:

1. **Conditional return modeling**
2. **Quadratic utility maximization**
3. **Convex portfolio optimization**
4. **Out-of-sample validation via walk-forward**

This framework belongs to the domain of:

- Empirical Asset Pricing
- Statistical Learning
- Quantitative Portfolio Construction

It combines classical financial theory with modern machine learning methods in a minimal and controlled implementation.

---

# 11. Project Structure (Implementation Layer)

```
portfolio/
  configs/
    config_data.yaml
    config_data.core25.yaml
    config_model.yaml
    config_backtest.yaml
    config_execution.yaml
  data/
    raw/
    processed/
  outputs/
    models/
    backtests/
  scripts/
    00_fetch_data.py
    01_build_panel.py
    02_train.py
    03_backtest.py
    04_rebalance.py
    05_report.py
    06_compare_portfolio_modes.py
    07_sweep_target_holding.py
    08_tune_market_neutral.py
    09_optimize_long_only_return.py
    10_compare_alpha_v2.py
    11_tune_alpha_v2_metrics.py
    12_optimize_drawdown_holdout.py
    13_paper_trading_cycle.py
  src/
    data.py
    features.py
    model_xgb.py
    risk.py
    optimizer.py
    backtest.py
    reporting.py
    execution/
      broker_base.py
      paper.py
      ibkr.py
  tests/
```

---

# 12. Environment Setup (.venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

For local secrets/config:

```bash
cp .env.example .env
```

---

# 13. Safety Defaults for IBKR Paper -> Production

- Default execution mode is `paper`.
- Use a separate `client_id` and account validation before sending orders.
- Keep `readonly=true` until execution checks are validated.
- Enforce hard limits (`max_position_weight`, `max_turnover_per_rebalance`, `kill_switch_enabled`) from `configs/config_execution.yaml`.
- Move to production only after walk-forward + paper trading checks are stable.

---

# 14. Rebalance Execution

Generate orders from latest backtest weights (dry-run by default):

```bash
python scripts/04_rebalance.py
```

Generate orders from latest `run_all` live recommendation (recommended for paper):

```bash
python scripts/04_rebalance.py \
  --weights-source run_all \
  --recommendation-path outputs/run_all/recommendation.json \
  --max-signal-age-business-days 3
```

Apply orders to configured broker:

```bash
python scripts/04_rebalance.py --apply
```

Generate factor diagnostics report:

```bash
python scripts/05_report.py
```

Key behavior:

- Reads target weights from `outputs/backtests/weights_history.parquet`
- Pulls account snapshot + prices from configured broker (`paper` or `ibkr`)
- Applies risk controls from `configs/config_execution.yaml`
- Saves order plan and execution summary in `outputs/execution/`

---

# 14.1 One-Shot Pipeline (`run_all`)

Run the full pipeline and get both:

- Backtest snapshot weights (latest OOS rebalance date).
- Live weights (latest market date with available features).

```bash
python scripts/05_run_all.py
```

Key outputs:

- `outputs/run_all/recommendation.json`
- `outputs/run_all/recommended_weights_<mode>_<live_date>.csv`
- `outputs/run_all/recommended_weights_backtest_<mode>_<rebalance_date>.csv`

In `recommendation.json`:

- `rebalance_date`: latest OOS backtest rebalance date.
- `live_signal_date`: latest live signal date.
- `artifacts.recommended_weights_csv`: live weights file to use for paper start/updates.

Market-context freshness (live mode):

- If `market_context` is a few business days behind, live inference applies a controlled fallback
  (forward-fill tail only) and reports it in `predict_live_summary`.
- Controls in `configs/config_data.yaml` -> `market_context`:
  - `live_allow_stale_fallback` (default `true`)
  - `live_max_stale_days` (default `3`)

---

# 14.2 Paper Trading Cycle (Serious Tests)

One command for operational paper tests:

1. Run full pipeline (`05_run_all.py`)
2. Rebalance from live recommendation (`04_rebalance.py --weights-source run_all`)
3. Save cycle report with artifacts and checks

Dry-run cycle:

```bash
python scripts/13_paper_trading_cycle.py
```

Apply to broker configured in `config_execution.yaml`:

```bash
python scripts/13_paper_trading_cycle.py --apply
```

Output report:

- `outputs/paper_cycle/paper_cycle_<timestamp>.json`

---

# 15. Return Boost Workflow (Before/After + Auto-Tuning)

Optimize long-only return with automated sweep on:

- Universe config
- Rebalance cadence
- Allocation method (`score_over_vol`, `mean_variance`)
- `weight_max`
- `max_turnover_per_rebalance`

Run:

```bash
python scripts/09_optimize_long_only_return.py \
  --baseline-config-data configs/config_data.core25.yaml \
  --baseline-config-backtest configs/config_backtest.market_neutral_tuned_exec.yaml \
  --baseline-config-execution configs/config_execution.market_neutral_tuned_exec.yaml \
  --candidate-config-data configs/config_data.yaml,configs/config_data.universe_expanded.yaml \
  --rebalance-days 10,15,20 \
  --allocation-methods score_over_vol,mean_variance \
  --weight-max-values 0.10,0.15,0.20 \
  --turnover-cap-values 0.25,0.35,0.45
```

Outputs:

- `outputs/experiments/return_boost/comparison.csv`
- `outputs/experiments/return_boost/report.json`
- `outputs/experiments/return_boost/config_data.recommended.yaml`
- `outputs/experiments/return_boost/config_backtest.recommended.yaml`
- `outputs/experiments/return_boost/config_execution.recommended.yaml`

---

# 16. Alpha V2 Package (Signal + Regime Gate)

Alpha V2 includes:

- Cross-sectional rank target transform during model training.
- Expanded feature set (momentum horizons, volatility structure, liquidity, microstructure).
- Signal-quality regime gate in backtest (de-risk when rolling signal quality deteriorates).

Baseline vs Alpha V2 comparison:

```bash
python scripts/10_compare_alpha_v2.py \
  --config-data configs/config_data.yaml \
  --config-execution configs/config_execution.yaml \
  --baseline-model configs/config_model.baseline_v1.yaml \
  --baseline-backtest configs/config_backtest.baseline_v1.yaml \
  --alpha-model configs/config_model.alpha_v2.yaml \
  --alpha-backtest configs/config_backtest.alpha_v2.yaml
```

Comparison report outputs:

- `outputs/experiments/alpha_v2_compare/comparison_report.json`
- `outputs/experiments/alpha_v2_compare/comparison_report.md`

---

# 17. Drawdown-First Holdout Tuning (Keep Sharpe)

Objective:

- Minimize drawdown while preserving Sharpe on a strict holdout window.

Default holdout:

- `2024-01-01` to latest available date.

Run:

```bash
python scripts/12_optimize_drawdown_holdout.py \
  --holdout-start 2024-01-01 \
  --max-weekly-sharpe-drop 0.0 \
  --max-sharpe-drop 0.0 \
  --apply-defaults
```

Outputs:

- `outputs/experiments/drawdown_holdout_tuning/report.json`
- `outputs/experiments/drawdown_holdout_tuning/structural_sweep.csv`
- `outputs/experiments/drawdown_holdout_tuning/overlay_sweep.csv`
- `outputs/experiments/drawdown_holdout_tuning/gate_sweep.csv`
- `outputs/experiments/drawdown_holdout_tuning/config_backtest.drawdown_holdout.tuned.yaml`
- `outputs/experiments/drawdown_holdout_tuning/config_execution.drawdown_holdout.tuned.yaml`
