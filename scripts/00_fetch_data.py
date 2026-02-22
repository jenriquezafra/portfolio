from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import run_fetch_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw market data and persist it to parquet.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_data.yaml"),
        help="Path to data config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    prices, output_path = run_fetch_data(config_path=config_path)

    print("Fetch complete")
    print(f"Rows: {len(prices):,}")
    print(f"Tickers: {prices['ticker'].nunique()}")
    print(f"Date range: {prices['date'].min().date()} -> {prices['date'].max().date()}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
