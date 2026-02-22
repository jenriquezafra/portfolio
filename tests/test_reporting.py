from __future__ import annotations

from src.reporting import build_factor_diagnostics_report


def test_factor_diagnostics_report_marks_pass_when_exposures_are_small() -> None:
    backtest_summary = {
        "portfolio_mode": "market_neutral",
        "allocation_method": "score_over_vol",
        "sharpe_ratio": 0.9,
        "weekly_sharpe_ratio": 1.1,
        "n_rebalances": 120,
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
    }
    factor_exposure_report = {
        "factor_names": ["nasdaq_proxy", "growth_proxy"],
        "ex_ante": {
            "mean": {"nasdaq_proxy": 0.01, "growth_proxy": -0.02},
            "max_abs": {"nasdaq_proxy": 0.12, "growth_proxy": 0.11},
        },
        "ex_post": {
            "full_sample": {
                "betas": {"nasdaq_proxy": 0.05, "growth_proxy": -0.04},
                "r2": 0.03,
            },
            "by_year": [
                {
                    "year": 2022,
                    "n_obs": 252,
                    "betas": {"nasdaq_proxy": 0.07, "growth_proxy": -0.03},
                },
                {
                    "year": 2023,
                    "n_obs": 252,
                    "betas": {"nasdaq_proxy": 0.03, "growth_proxy": -0.04},
                },
            ],
        },
    }

    report = build_factor_diagnostics_report(
        backtest_summary=backtest_summary,
        factor_exposure_report=factor_exposure_report,
    )

    assert report["status"] == "pass"
    assert report["checks"]["ex_post_beta_abs_ok"] is True
    assert report["checks"]["ex_post_r2_ok"] is True
    assert report["factor_names"] == ["nasdaq_proxy", "growth_proxy"]


def test_factor_diagnostics_report_marks_fail_when_exposures_are_large() -> None:
    report = build_factor_diagnostics_report(
        backtest_summary={},
        factor_exposure_report={
            "factor_names": ["nasdaq_proxy"],
            "ex_ante": {
                "mean": {"nasdaq_proxy": 0.30},
                "max_abs": {"nasdaq_proxy": 1.25},
            },
            "ex_post": {
                "full_sample": {
                    "betas": {"nasdaq_proxy": 0.42},
                    "r2": 0.51,
                },
                "by_year": [
                    {"year": 2024, "n_obs": 200, "betas": {"nasdaq_proxy": 0.55}},
                    {"year": 2025, "n_obs": 220, "betas": {"nasdaq_proxy": -0.48}},
                ],
            },
        },
    )

    assert report["status"] == "fail"
    assert report["checks"]["ex_ante_mean_abs_ok"] is False
    assert report["checks"]["ex_ante_max_abs_ok"] is False
    assert report["checks"]["ex_post_beta_abs_ok"] is False
    assert report["checks"]["ex_post_r2_ok"] is False
