from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_yaml


SUMMARY_FILENAME_RE = re.compile(r"^rebalance_\d{4}-\d{2}-\d{2}_.+_summary\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper trading cycle only when rebalance cadence is due "
            "(based on latest clean prices date and last applied rebalance)."
        )
    )
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-model", type=Path, default=Path("configs/config_model.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--max-signal-age-business-days",
        type=int,
        default=3,
        help="Same guardrail passed to scripts/14_paper_trading_cycle.py.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Same top-k passed to scripts/14_paper_trading_cycle.py.")
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=Path("outputs/execution"),
        help="Directory containing rebalance summaries.",
    )
    parser.add_argument(
        "--cycle-output-dir",
        type=Path,
        default=Path("outputs/paper_cycle"),
        help="Output dir for scripts/14_paper_trading_cycle.py reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/execution/auto_rebalance_latest.json"),
        help="Latest decision report path.",
    )
    parser.add_argument(
        "--skip-run-all",
        action="store_true",
        help="Pass --skip-run-all to scripts/14_paper_trading_cycle.py when execution is due.",
    )
    parser.add_argument("--apply", action="store_true", help="Send orders when due.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cycle execution even when cadence is not due.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _business_days_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        return 0
    return max(0, int(len(pd.bdate_range(start=start, end=end)) - 1))


def _next_rebalance_date(last_rebalance_date: pd.Timestamp, rebalance_every_n_days: int) -> pd.Timestamp:
    if rebalance_every_n_days <= 0:
        raise ValueError("`rebalance_every_n_days` must be positive.")
    dates = pd.bdate_range(start=pd.Timestamp(last_rebalance_date).normalize(), periods=rebalance_every_n_days + 1)
    return pd.Timestamp(dates[-1]).normalize()


def evaluate_schedule(
    *,
    latest_clean_date: pd.Timestamp,
    last_applied_rebalance_date: pd.Timestamp | None,
    rebalance_every_n_days: int,
    force: bool = False,
) -> dict[str, Any]:
    latest = pd.Timestamp(latest_clean_date).normalize()
    if rebalance_every_n_days <= 0:
        raise ValueError("`rebalance_every_n_days` must be positive.")

    if force:
        return {
            "due": True,
            "reason": "forced",
            "business_days_since_last_applied": None,
            "remaining_business_days": 0,
            "next_due_date": None,
        }

    if last_applied_rebalance_date is None:
        return {
            "due": True,
            "reason": "no_previous_applied_rebalance",
            "business_days_since_last_applied": None,
            "remaining_business_days": 0,
            "next_due_date": None,
        }

    last = pd.Timestamp(last_applied_rebalance_date).normalize()
    bdays = _business_days_between(last, latest)
    next_due = _next_rebalance_date(last, rebalance_every_n_days)
    due = bdays >= rebalance_every_n_days
    remaining = max(0, int(rebalance_every_n_days - bdays))
    return {
        "due": bool(due),
        "reason": "cadence_due" if due else "cadence_not_due",
        "business_days_since_last_applied": int(bdays),
        "remaining_business_days": int(remaining),
        "next_due_date": str(next_due.date()),
    }


def _load_latest_clean_date(config_data_path: Path) -> tuple[pd.Timestamp, Path]:
    config_data = load_yaml(config_data_path)
    data_section = config_data.get("data", {})
    if not isinstance(data_section, dict):
        raise ValueError("Missing `data` section in config_data.")
    clean_rel = data_section.get("output_clean_path")
    if not isinstance(clean_rel, str):
        raise ValueError("Missing `data.output_clean_path` in config_data.")

    clean_path = _resolve_path(Path(clean_rel))
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean prices file: {clean_path}")

    clean_dates = pd.read_parquet(clean_path, columns=["date"])
    clean_dates["date"] = pd.to_datetime(clean_dates["date"], utc=False).dt.tz_localize(None)
    latest = pd.Timestamp(clean_dates["date"].max()).normalize()
    return latest, clean_path


def _load_rebalance_every_n_days(config_backtest_path: Path) -> int:
    config_backtest = load_yaml(config_backtest_path)
    backtest = config_backtest.get("backtest", {})
    if not isinstance(backtest, dict):
        raise ValueError("Missing `backtest` section in config_backtest.")
    freq = str(backtest.get("rebalance_frequency", "every_n_days")).lower()
    if freq != "every_n_days":
        raise ValueError("This script currently supports only `rebalance_frequency=every_n_days`.")
    every_n = int(backtest.get("rebalance_every_n_days", 20))
    if every_n <= 0:
        raise ValueError("`backtest.rebalance_every_n_days` must be positive.")
    return every_n


def _list_rebalance_summary_paths(execution_dir: Path) -> list[Path]:
    candidates = sorted(execution_dir.glob("rebalance_*_summary.json"))
    return [p for p in candidates if SUMMARY_FILENAME_RE.match(p.name)]


def _load_last_applied_rebalance(execution_dir: Path) -> dict[str, Any] | None:
    paths = _list_rebalance_summary_paths(execution_dir)
    applied_rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("apply") is not True:
            continue
        ts_raw = payload.get("timestamp_utc")
        rb_raw = payload.get("rebalance_date")
        if ts_raw is None or rb_raw is None:
            continue
        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        rb = pd.to_datetime(rb_raw, utc=False, errors="coerce")
        if pd.isna(ts) or pd.isna(rb):
            continue
        applied_rows.append(
            {
                "summary_path": str(path),
                "timestamp_utc": pd.Timestamp(ts).isoformat(),
                "rebalance_date": str(pd.Timestamp(rb).date()),
                "broker": payload.get("broker"),
                "n_orders": payload.get("n_orders"),
            }
        )

    if not applied_rows:
        return None

    applied_rows = sorted(applied_rows, key=lambda row: row["timestamp_utc"])
    return applied_rows[-1]


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _latest_cycle_report(cycle_output_dir: Path) -> Path | None:
    reports = sorted(cycle_output_dir.glob("paper_cycle_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def main() -> None:
    args = parse_args()

    config_data = _resolve_path(args.config_data)
    config_model = _resolve_path(args.config_model)
    config_backtest = _resolve_path(args.config_backtest)
    config_execution = _resolve_path(args.config_execution)
    execution_dir = _resolve_path(args.execution_dir)
    cycle_output_dir = _resolve_path(args.cycle_output_dir)
    output_latest = _resolve_path(args.output)
    output_ts = output_latest.with_name(
        f"auto_rebalance_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )

    latest_clean_date, clean_path = _load_latest_clean_date(config_data)
    rebalance_every_n_days = _load_rebalance_every_n_days(config_backtest)
    last_applied = _load_last_applied_rebalance(execution_dir)
    last_applied_date = None if last_applied is None else pd.Timestamp(last_applied["rebalance_date"]).normalize()

    schedule = evaluate_schedule(
        latest_clean_date=latest_clean_date,
        last_applied_rebalance_date=last_applied_date,
        rebalance_every_n_days=rebalance_every_n_days,
        force=bool(args.force),
    )

    decision: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "latest_clean_prices_date": str(latest_clean_date.date()),
        "clean_prices_path": str(clean_path),
        "rebalance_every_n_days": int(rebalance_every_n_days),
        "last_applied_rebalance": last_applied,
        "schedule": schedule,
        "action": "skipped_not_due",
        "forced": bool(args.force),
        "apply": bool(args.apply),
        "skip_run_all": bool(args.skip_run_all),
        "cycle_report_path": None,
        "cycle_command": None,
    }

    if bool(schedule["due"]):
        cycle_script = _resolve_path(Path("scripts/14_paper_trading_cycle.py"))
        cmd = [
            sys.executable,
            str(cycle_script),
            "--config-data",
            str(config_data),
            "--config-model",
            str(config_model),
            "--config-backtest",
            str(config_backtest),
            "--config-execution",
            str(config_execution),
            "--max-signal-age-business-days",
            str(args.max_signal_age_business_days),
            "--top-k",
            str(args.top_k),
            "--output-dir",
            str(cycle_output_dir),
        ]
        if args.skip_run_all:
            cmd.append("--skip-run-all")
        if args.apply:
            cmd.append("--apply")

        decision["cycle_command"] = cmd
        decision["action"] = "executed"
        returncode, _, _ = _run_cmd(cmd, cwd=PROJECT_ROOT)
        if returncode != 0:
            decision["status"] = "error"
            decision["action"] = "execution_failed"
            output_latest.parent.mkdir(parents=True, exist_ok=True)
            output_latest.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
            output_ts.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
            raise RuntimeError(f"Auto-rebalance cycle failed with return code {returncode}.")

        cycle_report = _latest_cycle_report(cycle_output_dir)
        decision["cycle_report_path"] = None if cycle_report is None else str(cycle_report)

    output_latest.parent.mkdir(parents=True, exist_ok=True)
    output_latest.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    output_ts.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    print("Auto rebalance decision completed")
    print(f"Due now: {schedule.get('due')}")
    print(f"Reason: {schedule.get('reason')}")
    print(f"Latest clean prices date: {latest_clean_date.date()}")
    print(
        "Business days since last applied rebalance: "
        f"{schedule.get('business_days_since_last_applied')}"
    )
    print(f"Next due date: {schedule.get('next_due_date')}")
    print(f"Action: {decision.get('action')}")
    print(f"Latest report: {output_latest}")


if __name__ == "__main__":
    main()
