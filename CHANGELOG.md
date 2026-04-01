# Changelog

Registro de cambios relevantes del proyecto.

## 2026-03-03

### Añadido
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/17_execution_history.py` para consolidar histórico operativo de rebalances:
  - construye `outputs/execution/rebalance_history.csv`;
  - construye `outputs/execution/rebalance_history_summary.json`;
  - incluye métricas de equity/cash pre-rebalance, exposición objetivo, notional planificado y diagnóstico de posible infra-inversión.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/18_execution_dashboard.py` para monitor diario:
  - construye `outputs/execution/dashboard_daily.csv`;
  - construye `outputs/execution/dashboard_daily.png`.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/19_auto_rebalance_if_due.py` para automatizar rebalance por cadencia:
  - verifica si corresponde rebalance según días hábiles (`rebalance_every_n_days`);
  - ejecuta `scripts/14_paper_trading_cycle.py` solo si está vencido (o con `--force`);
  - guarda decisión en `outputs/execution/auto_rebalance_latest.json`.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_execution_history_report.py`.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_run_all_prev_weights_source.py`.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_execution_dashboard.py`.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_auto_rebalance_if_due.py`.

### Cambiado
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/06_run_all.py` ahora sólo usa `paper_state` como posiciones previas cuando `execution.broker=paper`, evitando que una configuración IBKR herede estado paper para turnover/live sizing.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_backtest.yaml` suaviza de-risk del `signal_quality_gate` para mantener mayor nivel de inversión:
  - `threshold: 0.0`
  - `bad_state_multiplier: 0.9`
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_execution.yaml` incrementa capacidad de despliegue:
  - `max_turnover_per_rebalance: 0.6`
  - `min_cash_buffer: 0.005`
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/notebooks/pipeline_operativo.ipynb` incorpora:
  - bloque de histórico + dashboard (`scripts/17` y `scripts/18`);
  - bloque de auto-rebalance por cadencia (`scripts/19`) con flags seguros por defecto.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/README.md` y `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/CLI_CHEATSHEET.md` documentan histórico, dashboard y auto-rebalance.

## 2026-02-26

### Añadido
- Nuevo módulo `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/src/signals.py` con señales cross-sectional determinísticas y sin look-ahead:
  - `momentum_residual_signal`
  - `reversal_regime_signal`
  - `vol_compression_breakout_signal`
  - `liquidity_impulse_signal`
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/16_tune_signal_stack.py` para tuning de `signal_stack` con búsqueda coarse + refinamiento local.
- Nuevo test `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_signal_stack.py`.

### Cambiado
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/src/backtest.py` ahora soporta `backtest.signal_stack`:
  - combinación de señal del modelo + señales ingenierizadas con pesos configurables;
  - columnas de atribución por rebalance en `rebalance_log.parquet`;
  - metadatos `signal_stack_*` en `backtest_summary.json`.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/06_run_all.py`:
  - aplica `signal_stack` también en cálculo de pesos live;
  - añade snapshot baseline opcional con `--snapshot-baseline`;
  - guarda bundle fechado con `comparison.json` + configs en `outputs/experiments/signal_stack_baseline/`.
- Se añadieron claves `backtest.signal_stack` en:
  - `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_backtest.yaml`
  - `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_backtest.alpha_v2.yaml`
  - `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/configs/config_backtest.baseline_v1.yaml`
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/tests/test_integration_backtest.py` incorpora cobertura para:
  - ejecución con `signal_stack.enabled=true`;
  - compatibilidad al desactivar el stack.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/README.md` documenta el flujo de signal stack y tuning.

## 2026-02-23

### Añadido
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/10_optimize_long_only_return.py` para comparar baseline vs candidatos.
- El script barre `rebalance_every_n_days`, `allocation_method`, `weight_max` y `max_turnover_per_rebalance`.
- El script genera `comparison.json/csv`, `report.json/md` y configs recomendadas.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/11_compare_alpha_v2.py` para comparar baseline vs Alpha v2 en una corrida reproducible.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/12_tune_alpha_v2_metrics.py` para tuning multietapa (modelo, estructura y gate) con score compuesto.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/13_optimize_drawdown_holdout.py` para optimización "drawdown-first" con constraint de Sharpe en holdout.
- Nuevo script `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/14_paper_trading_cycle.py` para ejecutar ciclo operativo de paper trading (`run_all` + `rebalance`) con reporte de sesión.
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
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/06_run_all.py` ahora exporta dos snapshots: `backtest` (último rebalance OOS) y `live` (última fecha de mercado disponible), manteniendo la selección de estrategia por backtest.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/04_rebalance.py` añade `--weights-source run_all` para rebalancear con pesos live de `recommendation.json`, con control de antigüedad de señal en días hábiles.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/04_rebalance.py` ahora también actualiza artefactos estables de última ejecución: `outputs/execution/rebalance_latest_orders.csv` y `outputs/execution/rebalance_latest_summary.json`.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/src/features.py` amplió el set de señales (horizontes de retorno, estructura de volatilidad, liquidez y microestructura) y añadió `drop_target_na` para soportar inferencia live en la cola sin label.
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/src/model_xgb.py` incorporó `training_target_transform` con opción `cross_sectional_rank`, añadió `run_predict_live`/`predict_latest_live_xgb` y fallback controlado de `market_context` en live (`live_allow_stale_fallback`, `live_max_stale_days`).
- `/Users/jenriquezafra/Proyectos/Dev/python/portfolio/scripts/14_paper_trading_cycle.py` ahora actualiza `outputs/paper_cycle/paper_cycle_latest.json` además del reporte con timestamp.
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
- Corrida completa validada (`scripts/06_run_all.py`):
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
