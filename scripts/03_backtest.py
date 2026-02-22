from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run portfolio backtest using OOS model predictions.")
    parser.add_argument(
        "--config-data",
        type=Path,
        default=Path("configs/config_data.yaml"),
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--config-backtest",
        type=Path,
        default=Path("configs/config_backtest.yaml"),
        help="Path to backtest config YAML.",
    )
    parser.add_argument(
        "--config-execution",
        type=Path,
        default=Path("configs/config_execution.yaml"),
        help="Path to execution/risk config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        daily_returns,
        weights_history,
        rebalance_log,
        summary,
        daily_path,
        weights_path,
        rebalance_log_path,
        summary_path,
    ) = run_backtest(
        config_data_path=args.config_data.resolve(),
        config_backtest_path=args.config_backtest.resolve(),
        config_execution_path=args.config_execution.resolve(),
    )

    print("Backtest complete")
    print(f"Daily rows: {len(daily_returns):,}")
    print(f"Rebalance events: {rebalance_log['rebalance_date'].nunique()}")
    print(f"Total return: {summary['total_return']:.6f}")
    print(f"Annualized return: {summary['annualized_return']:.6f}")
    print(f"Annualized volatility: {summary['annualized_volatility']:.6f}")
    print(f"Sharpe ratio: {summary['sharpe_ratio']}")
    print(f"Weekly Sharpe ratio: {summary.get('weekly_sharpe_ratio')}")
    print(f"Max drawdown: {summary['max_drawdown']:.6f}")
    print(f"Average turnover: {summary['average_turnover']:.6f}")
    print(f"Portfolio mode: {summary.get('portfolio_mode')}")
    print(f"Allocation method: {summary.get('allocation_method')}")
    print(f"Average gross exposure: {summary.get('average_gross_exposure')}")
    print(f"Average net exposure: {summary.get('average_net_exposure')}")
    print(f"Average ex-ante beta: {summary.get('average_ex_ante_beta_to_factor')}")
    print(f"Average ex-ante factor exposure: {summary.get('average_ex_ante_factor_exposure')}")
    print(f"Ex-post factor betas: {summary.get('ex_post_factor_betas')}")
    print(f"Daily returns output: {daily_path}")
    print(f"Weights output: {weights_path}")
    print(f"Rebalance log output: {rebalance_log_path}")
    print(f"Summary output: {summary_path}")
    print(f"Subperiod report output: {summary_path.parent / 'subperiod_report.json'}")
    print(f"Factor exposure report output: {summary_path.parent / 'factor_exposure_report.json'}")
    print(f"Factor diagnostics report output: {summary_path.parent / 'factor_diagnostics_report.json'}")
    if not weights_history.empty:
        latest_date = weights_history["rebalance_date"].max()
        latest = weights_history[weights_history["rebalance_date"] == latest_date]
        top = latest.sort_values("weight", ascending=False).head(5)
        print(f"Top weights at latest rebalance ({latest_date.date()}):")
        for _, row in top.iterrows():
            print(f"  - {row['ticker']}: {row['weight']:.4f}")


if __name__ == "__main__":
    main()
