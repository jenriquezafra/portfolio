from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Operational paper-trading cycle: run full signal pipeline and then rebalance from "
            "run_all live recommendation with staleness guardrails."
        )
    )
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-model", type=Path, default=Path("configs/config_model.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument("--recommendation-path", type=Path, default=Path("outputs/run_all/recommendation.json"))
    parser.add_argument(
        "--max-signal-age-business-days",
        type=int,
        default=3,
        help="Reject live recommendation if signal date is too old vs latest prices date.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top positions to print from run_all.")
    parser.add_argument(
        "--skip-run-all",
        action="store_true",
        help="Skip running 06_run_all.py and use existing recommendation file.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="If set, sends orders to broker in 04_rebalance.py (paper mode recommended).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_cycle"))
    return parser.parse_args()


def _run_cmd(cmd: list[str], workdir: Path) -> tuple[str, str]:
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout or "", proc.stderr or ""


def _extract_summary_path(output: str) -> Path | None:
    for line in output.splitlines():
        if line.strip().startswith("Summary file:"):
            raw = line.split("Summary file:", 1)[1].strip()
            if raw:
                return Path(raw)
    return None


def _latest_execution_summary(exec_dir: Path) -> Path | None:
    summaries = sorted(exec_dir.glob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return summaries[0] if summaries else None


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_data = args.config_data.resolve()
    config_model = args.config_model.resolve()
    config_backtest = args.config_backtest.resolve()
    config_execution = args.config_execution.resolve()
    recommendation_path = args.recommendation_path.resolve()

    if args.max_signal_age_business_days <= 0:
        raise ValueError("`--max-signal-age-business-days` must be positive.")
    if args.top_k <= 0:
        raise ValueError("`--top-k` must be positive.")

    started_at = datetime.now(timezone.utc)

    print("[1/3] Pipeline signal generation")
    if not args.skip_run_all:
        _run_cmd(
            [
                sys.executable,
                str((PROJECT_ROOT / "scripts/06_run_all.py").resolve()),
                "--config-data",
                str(config_data),
                "--config-model",
                str(config_model),
                "--config-backtest",
                str(config_backtest),
                "--config-execution",
                str(config_execution),
                "--top-k",
                str(args.top_k),
            ],
            workdir=PROJECT_ROOT,
        )
    else:
        print("Skipping run_all.py as requested.")

    print("[2/3] Rebalance from live recommendation")
    rebalance_cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts/04_rebalance.py").resolve()),
        "--config-data",
        str(config_data),
        "--config-backtest",
        str(config_backtest),
        "--config-execution",
        str(config_execution),
        "--weights-source",
        "run_all",
        "--recommendation-path",
        str(recommendation_path),
        "--max-signal-age-business-days",
        str(args.max_signal_age_business_days),
    ]
    if args.apply:
        rebalance_cmd.append("--apply")
    rebalance_stdout, _ = _run_cmd(rebalance_cmd, workdir=PROJECT_ROOT)

    print("[3/3] Persist cycle report")
    if not recommendation_path.exists():
        raise FileNotFoundError(f"Missing recommendation file after cycle: {recommendation_path}")
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))

    summary_path = _extract_summary_path(rebalance_stdout)
    if summary_path is None or not summary_path.exists():
        summary_path = _latest_execution_summary((PROJECT_ROOT / "outputs/execution").resolve())
    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError("Could not locate rebalance summary file.")
    rebalance_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_at_utc": started_at.isoformat(timespec="seconds"),
        "apply": bool(args.apply),
        "skip_run_all": bool(args.skip_run_all),
        "config_paths": {
            "config_data": str(config_data),
            "config_model": str(config_model),
            "config_backtest": str(config_backtest),
            "config_execution": str(config_execution),
            "recommendation_path": str(recommendation_path),
        },
        "recommendation": {
            "recommended_strategy": recommendation.get("recommended_strategy"),
            "rebalance_date": recommendation.get("rebalance_date"),
            "live_signal_date": recommendation.get("live_signal_date"),
            "recommended_weights_csv": recommendation.get("artifacts", {}).get("recommended_weights_csv"),
            "recommended_weights_backtest_csv": recommendation.get("artifacts", {}).get(
                "recommended_weights_backtest_csv"
            ),
        },
        "rebalance": {
            "summary_path": str(summary_path),
            "mode": rebalance_summary.get("mode"),
            "broker": rebalance_summary.get("broker"),
            "apply": rebalance_summary.get("apply"),
            "rebalance_date_used": rebalance_summary.get("rebalance_date"),
            "n_orders": rebalance_summary.get("n_orders"),
            "portfolio_turnover_estimate": rebalance_summary.get("portfolio_turnover_estimate"),
            "weights_source_meta": rebalance_summary.get("weights_source_meta"),
            "orders_file": rebalance_summary.get("orders_file"),
        },
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"paper_cycle_{ts}.json"
    latest_report_path = output_dir / "paper_cycle_latest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    latest_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("Paper trading cycle completed")
    print(f"Recommendation live signal date: {report['recommendation']['live_signal_date']}")
    print(f"Rebalance summary: {summary_path}")
    print(f"Cycle report: {report_path}")
    print(f"Latest cycle report: {latest_report_path}")


if __name__ == "__main__":
    main()
