# Changelog

Registro de cambios relevantes del proyecto.

## 2026-02-23

### Añadido
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/09_optimize_long_only_return.py` para comparar baseline vs candidatos.
- El script barre `rebalance_every_n_days`, `allocation_method`, `weight_max` y `max_turnover_per_rebalance`.
- El script genera `comparison.json/csv`, `report.json/md` y configs recomendadas.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/10_compare_alpha_v2.py` para comparar baseline vs Alpha v2 en una corrida reproducible.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/11_tune_alpha_v2_metrics.py` para tuning multietapa (modelo, estructura y gate) con score compuesto.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/12_optimize_drawdown_holdout.py` para optimización "drawdown-first" con constraint de Sharpe en holdout.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/13_paper_trading_cycle.py` para ejecutar ciclo operativo de paper trading (`run_all` + `rebalance`) con reporte de sesión.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_model_target_transform.py`.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_model_live_predict.py`.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_rebalance_weights_source.py`.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_signal_quality_gate.py`.
- Nuevo config `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_model.baseline_v1.yaml`.
- Nuevo config `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_model.alpha_v2.yaml`.
- Nuevo config `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_backtest.baseline_v1.yaml`.
- Nuevo config `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_backtest.alpha_v2.yaml`.
- Nuevo archivo `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_data.core25.yaml` para conservar el universo original de 25 activos.

### Cambiado
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_data.yaml` ahora usa el universo expandido.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_model.yaml` usa `training_target_transform: cross_sectional_rank`.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_backtest.yaml` quedó afinado en long-only con `rebalance_every_n_days: 20`, `weight_max: 0.2`, `risk_aversion_lambda: 20.0` y `signal_quality_gate` activo (`threshold: 0.02`, `bad_state_multiplier: 0.6`, `lookback_rebalances: 20`, `min_history_rebalances: 8`).
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_execution.yaml` fijó `max_turnover_per_rebalance: 0.35`.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/05_run_all.py` ahora exporta dos snapshots: `backtest` (último rebalance OOS) y `live` (última fecha de mercado disponible), manteniendo la selección de estrategia por backtest.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/04_rebalance.py` añade `--weights-source run_all` para rebalancear con pesos live de `recommendation.json`, con control de antigüedad de señal en días hábiles.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/src/features.py` amplió el set de señales (horizontes de retorno, estructura de volatilidad, liquidez y microestructura) y añadió `drop_target_na` para soportar inferencia live en la cola sin label.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/src/model_xgb.py` incorporó `training_target_transform` con opción `cross_sectional_rank`, añadió `run_predict_live`/`predict_latest_live_xgb` y fallback controlado de `market_context` en live (`live_allow_stale_fallback`, `live_max_stale_days`).
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/src/backtest.py` incorporó `signal_quality_gate` con control de exposición por calidad de señal histórica.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/README.md` documenta el flujo de optimización de retorno.

### Corregido
- Se corrigió el ticker `"ON"` en YAML (evitando parseo booleano) en `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_data.yaml`.
- Se corrigió el ticker `"ON"` en YAML (evitando parseo booleano) en `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_data.universe_expanded.yaml`.

### Resultados generados
- Corrida de referencia rápida en `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/experiments/return_boost_quick/`.
- Reporte before/after: `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/experiments/return_boost_quick/report.json`.
- Comparativa Alpha v2: `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/experiments/alpha_v2_compare_quick/comparison_report.json`.
- Tuning Alpha v2 (versión corregida): `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/experiments/alpha_v2_tuning_auto_v2/report.json`.
- Tuning de drawdown con holdout estricto: `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/experiments/drawdown_holdout_tuning/report.json`.
- Corrida completa validada (`scripts/05_run_all.py`):
  - `long_only`: `weekly_sharpe=1.5604`, `max_dd=-0.3544`, `score=1.2859`
  - `market_neutral`: `weekly_sharpe=0.9707`, `max_dd=-0.0904`, `score=0.8807`
  - Estrategia recomendada: `long_only`
  - Fecha de rebalanceo (snapshot backtest): `2026-01-14`
  - Fecha de señal live: `2026-02-23`
  - Frescura de market context en live: `stale_business_days=1` con `fallback_applied=true`.
  - Weights live recomendados: `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/run_all/recommended_weights_long_only_2026-02-23.csv`
  - Weights snapshot backtest: `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/run_all/recommended_weights_backtest_long_only_2026-01-14.csv`
  - Reporte de recomendación: `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/outputs/run_all/recommendation.json`
- Tests ejecutados: `19 passed`.

### Nota operativa
- En entorno sandbox sin red, Yahoo puede fallar por DNS (`Could not resolve host: guce.yahoo.com`); la validación final se ejecutó con permisos de red.
