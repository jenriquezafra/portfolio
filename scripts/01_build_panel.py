from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import run_build_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw prices, create features/labels panel, and persist processed datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_data.yaml"),
        help="Path to data config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_df, panel_df, clean_path, panel_path = run_build_panel(config_path=args.config.resolve())

    print("Build panel complete")
    print(f"Clean rows: {len(clean_df):,}")
    print(f"Clean tickers: {clean_df['ticker'].nunique()}")
    print(f"Clean date range: {clean_df['date'].min().date()} -> {clean_df['date'].max().date()}")
    print(f"Clean output: {clean_path}")
    print(f"Panel rows: {len(panel_df):,}")
    print(f"Panel tickers: {panel_df['ticker'].nunique()}")
    print(f"Panel date range: {panel_df['date'].min().date()} -> {panel_df['date'].max().date()}")
    print(f"Panel output: {panel_path}")


if __name__ == "__main__":
    main()
