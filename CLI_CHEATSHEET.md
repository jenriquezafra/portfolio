# CLI Cheatsheet (Operativo)

Usa siempre desde la raíz del repo:

```bash
cd /Users/jenriquezafra/Proyectos/Dev/python/portfolio
PY=.venv/bin/python
```

## Flujo diario (recomendado)

1) Ejecutar pipeline completo y generar recomendación live:

```bash
$PY scripts/06_run_all.py
```

2) Previsualizar rebalance (sin enviar órdenes):

```bash
$PY scripts/04_rebalance.py \
  --weights-source run_all \
  --recommendation-path outputs/run_all/recommendation.json \
  --max-signal-age-business-days 3
```

3) Enviar órdenes (apply):

```bash
$PY scripts/04_rebalance.py \
  --weights-source run_all \
  --recommendation-path outputs/run_all/recommendation.json \
  --max-signal-age-business-days 3 \
  --apply
```

Nota: en `configs/config_execution.yaml` ya está por defecto `order_type: LMT` y `tif: GTC`.

## Operación IBKR

Chequeo de conexión:

```bash
$PY scripts/15_check_ibkr_connection.py --symbols SPY,QQQ
```

Chequeo de estado de órdenes (IDs del último rebalance):

```bash
$PY scripts/check_orders.py
```

Ciclo completo en un solo comando (`run_all + rebalance`):

```bash
$PY scripts/14_paper_trading_cycle.py
```

Con envío de órdenes:

```bash
$PY scripts/14_paper_trading_cycle.py --apply
```

## Diagnóstico y reportes

Reporte de factores:

```bash
$PY scripts/05_report.py
```

Histórico operativo de rebalances (equity/cash/exposición objetivo):

```bash
$PY scripts/17_execution_history.py
```

Dashboard diario (tabla + gráfico):

```bash
$PY scripts/18_execution_dashboard.py
```

Auto rebalance por cadencia (solo ejecuta ciclo si ya toca):

```bash
$PY scripts/19_auto_rebalance_if_due.py
```

Auto rebalance con envío de órdenes cuando toque:

```bash
$PY scripts/19_auto_rebalance_if_due.py --apply
```

## Archivos de salida que mirar primero

- `outputs/run_all/recommendation.json`
- `outputs/execution/rebalance_latest_orders.csv`
- `outputs/execution/rebalance_latest_summary.json`
- `outputs/execution/rebalance_history.csv`
- `outputs/execution/rebalance_history_summary.json`
- `outputs/execution/dashboard_daily.csv`
- `outputs/execution/dashboard_daily.png`
- `outputs/execution/auto_rebalance_latest.json`
- `outputs/paper_cycle/paper_cycle_latest.json`
