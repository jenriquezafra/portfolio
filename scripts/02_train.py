from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_xgb import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward XGBoost training and persist OOS predictions.")
    parser.add_argument(
        "--config-data",
        type=Path,
        default=Path("configs/config_data.yaml"),
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--config-model",
        type=Path,
        default=Path("configs/config_model.yaml"),
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--config-backtest",
        type=Path,
        default=Path("configs/config_backtest.yaml"),
        help="Path to backtest config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        predictions,
        training_log,
        importances,
        summary,
        predictions_path,
        log_path,
        importance_path,
        summary_path,
    ) = run_train(
        config_data_path=args.config_data.resolve(),
        config_model_path=args.config_model.resolve(),
        config_backtest_path=args.config_backtest.resolve(),
    )

    print("Training complete")
    print(f"Predictions rows: {len(predictions):,}")
    print(f"Rebalance dates: {training_log['rebalance_date'].nunique()}")
    print(f"OOS IC (Spearman): {summary['oos_ic_spearman']}")
    print(f"OOS CS-IC mean: {summary.get('oos_cs_ic_mean')}")
    print(f"OOS ICIR: {summary.get('oos_cs_ic_ir')}")
    print(f"OOS top-bottom mean: {summary.get('oos_top_bottom_mean')}")
    print(f"OOS top-bottom t-stat: {summary.get('oos_top_bottom_tstat')}")
    print(f"Validation IC mean (purged): {summary.get('avg_validation_ic_spearman')}")
    print(f"Predictions output: {predictions_path}")
    print(f"Training log output: {log_path}")
    print(f"Feature importance output: {importance_path}")
    print(f"Summary output: {summary_path}")
    if not importances.empty:
        top = (
            importances.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .head(5)
        )
        print("Top features (mean importance):")
        for _, row in top.iterrows():
            print(f"  - {row['feature']}: {row['importance']:.6f}")


if __name__ == "__main__":
    main()
