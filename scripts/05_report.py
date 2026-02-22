from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting import run_factor_diagnostics_from_outputs


def _parse_thresholds(raw: str | None) -> dict[str, float] | None:
    if raw is None or not raw.strip():
        return None
    out: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid threshold token: {token!r}. Expected key=value.")
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid threshold key in token: {token!r}")
        out[key] = float(value.strip())
    return out or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build automatic factor diagnostics from backtest outputs.")
    parser.add_argument(
        "--outputs-subdir",
        type=str,
        default="outputs/backtests",
        help="Project-relative output directory containing backtest_summary and factor_exposure_report.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help=(
            "Optional override thresholds as comma-separated key=value "
            "(e.g. ex_post_beta_abs_max=0.12,ex_post_r2_max=0.10)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds)

    report, path = run_factor_diagnostics_from_outputs(
        project_root=PROJECT_ROOT,
        output_subdir=args.outputs_subdir,
        thresholds=thresholds,
    )
    print(f"Factor diagnostics saved: {path}")
    print(f"Status: {report.get('status')}")
    print(f"Checks: {report.get('checks')}")
    recs = report.get("recommendations") or []
    if recs:
        print("Recommendations:")
        for rec in recs:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
